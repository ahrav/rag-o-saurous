package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/chewxy/math32"
	_ "github.com/mattn/go-sqlite3"
	"github.com/sashabaranov/go-openai"
)

func main() {
	dbPath := flag.String("db", "chunks.db", "path to DB with chunks")
	doCalculate := flag.Bool("calculate", false, "calculate embeddings and update DB")
	clearEmbeddings := flag.Bool("clear", false, "clear embeddings in DB")
	question := flag.String("question", "", "question to answer")
	doAnswer := flag.Bool("answer", false, "answer question (DB must have embeddings already)")
	flag.Parse()

	if *doAnswer && *question == "" {
		log.Fatal("must specify question to answer")
	}

	client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	if *doAnswer {
		answerQuestion(client, *question, *dbPath)
	} else if *doCalculate {
		calculateEmbeddings(client, *dbPath, *clearEmbeddings)
	}
}

// answerQuestion queries the database for chunk content and embeddings,
// scores each chunk against the question embedding with cosine similarity,
// takes the top scoring chunks as context, builds a prompt with the context
// and question, calls the OpenAI API to generate an answer, and prints the response.
func answerQuestion(client *openai.Client, question, dbPath string) {
	// Connect to the SQLite database
	db, err := sql.Open("sqlite3", dbPath)
	checkErr(err)
	defer db.Close()

	// SQL query to extract chunks' content along with embeddings.
	stmt, err := db.Prepare(`
		SELECT chunks.path, chunks.content, embeddings.embedding
		FROM chunks
		INNER JOIN embeddings
		ON chunks.id = embeddings.chunk_id
	`)
	checkErr(err)
	defer stmt.Close()

	rows, err := stmt.Query()
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	type scoreRecord struct {
		Path    string
		Score   float32
		Content string
	}
	var scores []scoreRecord

	// Iterate through the rows, scoring each chunk with cosine similarity to the question's embedding.
	qEmb := getEmbedding(client, question)
	for rows.Next() {
		var (
			path      string
			content   string
			embedding []byte
		)

		err = rows.Scan(&path, &content, &embedding)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("path: %s, content: %d, embedding: %d\n", path, len(content), len(embedding))

		contentEmb := decodeEmbedding(embedding)
		score := cosineSimilarity(qEmb, contentEmb)
		scores = append(scores, scoreRecord{path, score, content})
		fmt.Println(path, score)
	}
	if err = rows.Err(); err != nil {
		log.Fatal(err)
	}

	fmt.Println(len(scores))
	slices.SortFunc(scores, func(a, b scoreRecord) int {
		// The scores are in the range [0, 1], so scale them to get non-zero integers for comparison.
		return int(100.0 * (a.Score - b.Score))
	})

	// Take the 5 best-scoring chunks as context and paste them together into contextInfo.
	var contextInfo string
	for i := len(scores) - 1; i > len(scores)-6; i-- {
		contextInfo = contextInfo + "\n" + scores[i].Content
	}

	// Build the prompt and execute the LLM API.
	query := fmt.Sprintf(`Use the below information to answer the subsequent question.
Information:
%v

Question: %v`, contextInfo, question)

	fmt.Println("Query:\n", query)
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4TurboPreview,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: query,
				},
			},
		},
	)
	checkErr(err)

	fmt.Println("Got response, ID:", resp.ID)
	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	choice := resp.Choices[0]
	fmt.Println(choice.Message.Content)
}

// calculateEmbeddings concurrently generates embeddings for all chunks in the
// database that don't already have embeddings, using multiple goroutines.
// It uses a buffered channel to distribute work and collect results.
func calculateEmbeddings(client *openai.Client, dbPath string, clear bool) {
	db, err := sql.Open("sqlite3", dbPath)
	checkErr(err)
	defer db.Close()

	log.Println("Creating embeddings table if needed")
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS embeddings (
  		id INTEGER PRIMARY KEY AUTOINCREMENT,
    	chunk_id INTEGER,
    	embedding BLOB,
    	FOREIGN KEY (chunk_id) REFERENCES chunks(id)
		);`)
	checkErr(err)

	_, err = db.Exec("PRAGMA journal_mode=WAL;")
	checkErr(err)

	if clear {
		log.Println("Clearing embeddings table")
		_, err = db.Exec(`DELETE FROM embeddings`)
		checkErr(err)
	}

	type embData struct {
		chunkID int
		data    []byte
	}

	// Prepare for concurrent embedding generation.
	embChan := make(chan embData, 100)
	errChan := make(chan error, 1)
	done := make(chan struct{})
	var wg sync.WaitGroup

	go func() {
		// Step 2: insert all embedding data into the embeddings table.
		for emb := range embChan {
			res, err := db.Exec("INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)", emb.chunkID, emb.data)
			if err != nil {
				errChan <- err
				return
			}
			rowID, err := res.LastInsertId()
			checkErr(err)
			fmt.Println("Inserted into embeddings, id", rowID)
		}
		close(done)
	}()

	rows, err := db.Query(`
        SELECT c.id, c.path, c.nchunk, c.content
        FROM chunks c
        LEFT JOIN embeddings e ON c.id = e.chunk_id
        WHERE e.id IS NULL
  `)
	checkErr(err)
	defer rows.Close()

	maxWorkers := runtime.NumCPU()
	semaphore := make(chan struct{}, maxWorkers)
	// Step 1: calculate embeddings for all chunks in the DB, storing them in embs.
	for rows.Next() {
		var (
			id      int
			path    string
			nchunk  int
			content string
		)
		if err = rows.Scan(&id, &path, &nchunk, &content); err != nil {
			close(embChan)
			log.Fatal(err)
		}

		wg.Add(1)
		semaphore <- struct{}{}
		go func(id int, content, path string, nchunk int) {
			defer wg.Done()
			defer func() { <-semaphore }()
			// Check for previous errors to avoid unnecessary work.
			select {
			case <-errChan:
				return
			default:
				fmt.Printf("id: %d, path: %s, nchunk: %d, content: %d\n", id, path, nchunk, len(content))
				if len(content) > 0 {
					embeddings := getEmbeddings(client, content)
					for _, emb := range embeddings {
						emb := encodeEmbedding(emb)
						select {
						case embChan <- embData{chunkID: id, data: emb}:
						case err := <-errChan:
							log.Println("Error received, stopping:", err)
							return
						}
					}
				}
			}
		}(id, content, path, nchunk)
	}

	if err = rows.Err(); err != nil {
		log.Fatal(err)
	}

	wg.Wait()
	close(embChan)

	select {
	case err := <-errChan:
		log.Fatal("Operation failed:", err)
	default:
		<-done // Wait for the insertion routine to finish
	}

}

// getEmbedding invokes the OpenAI embedding API to calculate the embedding
// for the given string. It returns the embedding.
func getEmbeddings(client *openai.Client, data string) [][]float32 {
	var embeddings [][]float32
	const maxTokensPerRequest = 8192
	if len(data) <= maxTokensPerRequest {
		embeddings = append(embeddings, getEmbedding(client, data))
		return embeddings
	}

	chunks := splitIntoChunks(data, maxTokensPerRequest)
	for _, chunk := range chunks {
		embeddings = append(embeddings, getEmbedding(client, chunk))
	}
	return embeddings
}

func splitIntoChunks(data string, maxChars int) []string {
	var chunks []string
	var builder strings.Builder

	words := strings.Fields(data)

	for _, word := range words {
		// If adding the next word exceeds the maxChars limit,
		// or if adding the current word would exactly match the maxChars (hence, the +1 for space is accounted for),
		// append the current chunk to chunks and reset the builder.
		if builder.Len()+len(word)+1 > maxChars {
			if builder.Len() > 0 { // Ensure we don't append empty strings
				chunks = append(chunks, builder.String())
				builder.Reset()
			}
		}

		if builder.Len() > 0 {
			// Only add a space if the builder is not empty
			builder.WriteRune(' ')
		}
		builder.WriteString(word)
	}

	// Add the last chunk if it's not empty
	if builder.Len() > 0 {
		chunks = append(chunks, builder.String())
	}

	return chunks
}

func getEmbedding(client *openai.Client, data string) []float32 {
	queryReq := openai.EmbeddingRequest{
		Input: []string{data},
		Model: openai.LargeEmbedding3,
	}

	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	checkErr(err)
	return queryResponse.Data[0].Embedding
}

// encodeEmbedding encodes an embedding into a byte buffer, e.g. for DB storage as a blob.
func encodeEmbedding(emb []float32) []byte {
	buf := make([]byte, len(emb)*4)
	err := binary.Write(bytes.NewBuffer(buf), binary.LittleEndian, emb)
	checkErr(err)
	return buf
}

// decodeEmbedding decodes an embedding back from a byte buffer.
func decodeEmbedding(b []byte) []float32 {
	// Calculate how many float32 values are in the slice.
	count := len(b) / 4
	numbers := make([]float32, count)
	err := binary.Read(bytes.NewBuffer(b), binary.LittleEndian, numbers)
	checkErr(err)

	return numbers
}

// cosineSimilarity calculates cosine similarity (magnitude-adjusted dot product)
// between two vectors that must be of the same size.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("different lengths")
	}

	var aMag, bMag, dotProduct float32
	for i := range a {
		aMag += a[i] * a[i]
		bMag += b[i] * b[i]
		dotProduct += a[i] * b[i]
	}
	return dotProduct / (math32.Sqrt(aMag) * math32.Sqrt(bMag))
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}
