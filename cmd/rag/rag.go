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
	"strings"
	"sync"
	"time"

	"github.com/chewxy/math32"
	_ "github.com/mattn/go-sqlite3"
	mvclient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/sashabaranov/go-openai"
)

func main() {
	dbPath := flag.String("db", "chunks.db", "path to DB with chunks")
	doCalculate := flag.Bool("calculate", false, "calculate embeddings and update DB")
	question := flag.String("question", "", "question to answer")
	doAnswer := flag.Bool("answer", false, "answer question (DB must have embeddings already)")
	flag.Parse()

	if *doAnswer && *question == "" {
		log.Fatal("must specify question to answer")
	}

	ctx := context.Background()
	client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	mvClient, err := mvclient.NewClient(ctx, mvclient.Config{
		Address: "https://in03-26de6b5e6ea1fee.api.gcp-us-west1.zillizcloud.com",
		APIKey:  os.Getenv("MILVUS_API_KEY"),
	})
	checkErr(err)
	defer mvClient.Close()

	if *doAnswer {
		if err = mvClient.LoadCollection(ctx, collectionName, false); err != nil {
			log.Fatal("fail to load collection:", err.Error())
		}

		fmt.Println("Collection loaded")
		answerQuestion(client, mvClient, *question, *dbPath)
	} else if *doCalculate {
		calculateEmbeddings(client, mvClient, *dbPath)
	}
}

const collectionName = "test_collection"

// answerQuestion queries the database for chunk content and embeddings,
// scores each chunk against the question embedding with cosine similarity,
// takes the top scoring chunks as context, builds a prompt with the context
// and question, calls the OpenAI API to generate an answer, and prints the response.
func answerQuestion(client *openai.Client, mvClient mvclient.Client, question, dbPath string) {
	qEmb := getEmbedding(client, question)
	// Create a float vector from the question embedding.
	vector := entity.FloatVector(qEmb)

	// TODO: Look into this some more.
	// Use flat search param.
	sp, _ := entity.NewIndexFlatSearchParam()

	ctx := context.Background()
	sr, err := mvClient.Search(ctx, collectionName, nil, "", []string{"ChunkID"}, []entity.Vector{vector}, "Vector",
		entity.L2, 5, sp)
	checkErr(err)

	var contextInfo string
	for _, result := range sr {
		var idColumn *entity.ColumnInt64
		for _, field := range result.Fields {
			if field.Name() == "ChunkID" {
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					idColumn = c
				}
			}
		}

		if idColumn == nil {
			log.Fatal("result field not match")
		}

		for i := 0; i < result.ResultCount; i++ {
			id, err := idColumn.ValueByIdx(i)
			checkErr(err)

			// Retrieve the chunk content by chunk id.
			content, err := getContentByChunkID(id)
			checkErr(err)

			fmt.Printf("chunk id: %d scores: %f\n", id, result.Scores[i])
			contextInfo += fmt.Sprintf("Chunk ID: %d\nContent: %s\n\n", id, content)
		}
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
	if err != nil {
		log.Fatal("fail to create chat completion:", err.Error())
	}

	fmt.Println("Got response, ID:", resp.ID)
	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	choice := resp.Choices[0]
	fmt.Println(choice.Message.Content)
}

func getContentByChunkID(id int64) (string, error) {
	// Connect to the SQLite database
	db, err := sql.Open("sqlite3", "chunks.db")
	checkErr(err)
	defer db.Close()

	// SQL query to extract chunks' content along with embeddings.
	stmt, err := db.Prepare(`
		SELECT content
		FROM chunks
		WHERE id =?
	`)
	checkErr(err)
	defer stmt.Close()

	var content string
	err = stmt.QueryRow(id).Scan(&content)
	if err != nil {
		return "", err
	}

	return content, nil
}

// calculateEmbeddings concurrently generates embeddings for all chunks in the
// database that don't already have embeddings, using multiple goroutines.
// It uses a buffered channel to distribute work and collect results.
func calculateEmbeddings(client *openai.Client, mvClient mvclient.Client, dbPath string) {
	db, err := sql.Open("sqlite3", dbPath)
	checkErr(err)
	defer db.Close()

	_, err = db.Exec("PRAGMA journal_mode=WAL;")
	checkErr(err)

	type embData struct {
		chunkID int
		data    []float32
	}

	// Prepare for concurrent embedding generation.
	embChan := make(chan embData, 100)
	errChan := make(chan error, 1)
	done := make(chan struct{})
	var wg sync.WaitGroup

	go func() {
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
			defer cancel()
			if err = mvClient.Flush(ctx, collectionName, false); err != nil {
				log.Fatal("fail to flush:", err.Error())
			}

			fmt.Println("flush completed")

			idx, err := entity.NewIndexIvfFlat(entity.L2, 2)
			if err != nil {
				log.Fatal("fail to create ivf flat index:", err.Error())
			}
			if err = mvClient.CreateIndex(ctx, collectionName, "Vector", idx, false); err != nil {
				log.Fatal("fail to create index:", err.Error())
			}

			sidx := entity.NewScalarIndex()
			if err = mvClient.CreateIndex(ctx, collectionName, "ChunkID", sidx, false); err != nil {
				log.Fatal("fail to create index:", err.Error())
			}

			fmt.Println("index created")

			close(done)
		}()

		ctx := context.Background()
		has, err := mvClient.HasCollection(ctx, collectionName)
		if err != nil {
			errChan <- err
			return
		}

		if !has {
			schema := entity.NewSchema().WithName(collectionName).WithDescription("test collection").
				WithField(entity.NewField().WithName("ChunkID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
				WithField(entity.NewField().WithName("Vector").WithDataType(entity.FieldTypeFloatVector).WithDim(1536))

			if err = mvClient.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
				errChan <- err
				return
			}

			fmt.Println("Collection created")
		}

		// Step 2: insert all embedding data into the embeddings table.
		for emb := range embChan {
			vectorColumn := entity.NewColumnFloatVector("Vector", 1536, [][]float32{emb.data})
			_, err = mvClient.Insert(ctx, collectionName, "", entity.NewColumnInt64("ChunkID", []int64{int64(emb.chunkID)}), vectorColumn)
			if err != nil {
				errChan <- err
				return
			}

			fmt.Println("Inserted into embeddings, chunkID", emb.chunkID)
		}
	}()

	rows, err := db.Query(`SELECT id, path, nchunk, content FROM chunks WHERE id > 6200;`)
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
		Model: openai.SmallEmbedding3,
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
