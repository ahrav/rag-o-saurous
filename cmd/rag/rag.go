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

	_ "github.com/mattn/go-sqlite3"
	mvclient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func main() {
	dbPath := flag.String("db", "chunks.db", "path to DB with chunks")
	doCalculate := flag.Bool("calculate", false, "calculate embeddings and update DB")
	question := flag.String("question", "", "question to answer")
	doAnswer := flag.Bool("answer", false, "answer question (DB must have embeddings already)")
	useLocal := flag.Bool("local", false, "use local model")
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
	if err != nil {
		log.Fatal("fail to create milvus client:", err.Error())
	}
	defer mvClient.Close()

	switch {
	case *doAnswer:
		if err = mvClient.LoadCollection(ctx, collectionName, false); err != nil {
			log.Fatal("fail to load collection:", err.Error())
		}
		if err = answerQuestion(ctx, client, mvClient, *question, *useLocal); err != nil {
			log.Fatal("fail to answer question:", err.Error())
		}
	case *doCalculate:
		if err = calculateEmbeddings(ctx, client, mvClient, *dbPath); err != nil {
			log.Fatal("fail to calculate embeddings:", err.Error())
		}
	default:
		log.Fatal("must specify -calculate or -answer")
	}
}

const collectionName = "test_collection"

// answerQuestion queries the database for chunk content and embeddings,
// scores each chunk against the question embedding with cosine similarity,
// takes the top scoring chunks as context, builds a prompt with the context
// and question, calls the OpenAI API to generate an answer, and prints the response.
func answerQuestion(ctx context.Context, client *openai.Client, mvClient mvclient.Client, question string, useLocal bool) error {
	qEmb, err := getEmbedding(client, question)
	if err != nil {
		return fmt.Errorf("fail to get question embedding: %w", err)
	}
	// Create a float vector from the question embedding.
	vector := entity.FloatVector(qEmb)

	// TODO: Look into this some more.
	// Use flat search param.
	sp, _ := entity.NewIndexFlatSearchParam()

	sr, err := mvClient.Search(ctx, collectionName, nil, "", []string{"ChunkID"}, []entity.Vector{vector}, "Vector",
		entity.L2, 3, sp)
	if err != nil {
		return fmt.Errorf("fail to search db: %w", err)
	}

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
			return fmt.Errorf("fail to find chunk id column")
		}

		for i := 0; i < result.ResultCount; i++ {
			id, err := idColumn.ValueByIdx(i)
			if err != nil {
				return fmt.Errorf("fail to get value by idx: %w", err)
			}

			// Retrieve the chunk content by chunk id.
			content, err := getContentByChunkID(id)
			if err != nil {
				return fmt.Errorf("fail to get content by chunk id: %w", err)
			}

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

	if useLocal {
		llm, err := ollama.New(ollama.WithModel("llama2"))
		if err != nil {
			return fmt.Errorf("fail to create llama model: %w", err)
		}

		completion, err := llms.GenerateFromSinglePrompt(ctx, llm, query)
		if err != nil {
			return fmt.Errorf("fail to generate completion: %w", err)
		}

		fmt.Println(completion)

		return nil
	}

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
	b, err := json.MarshalIndent(resp.Choices[0], "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	choice := resp.Choices[0]
	fmt.Println(choice.Message.Content)
	return nil
}

func getContentByChunkID(id int64) (string, error) {
	// Connect to the SQLite database
	db, err := sql.Open("sqlite3", "chunks.db")
	if err != nil {
		return "", err
	}
	defer db.Close()

	// SQL query to extract chunks' content along with embeddings.
	stmt, err := db.Prepare(`
		SELECT content
		FROM chunks
		WHERE id =?
	`)
	if err != nil {
		return "", err
	}
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
func calculateEmbeddings(ctx context.Context, client *openai.Client, mvClient mvclient.Client, dbPath string) error {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return fmt.Errorf("fail to open db: %w", err)
	}
	defer db.Close()

	if _, err = db.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		return fmt.Errorf("fail to set journal mode: %w", err)
	}

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
			ctx, cancel := context.WithTimeout(ctx, time.Second*30)
			defer cancel()
			if err = mvClient.Flush(ctx, collectionName, false); err != nil {
				log.Fatal("fail to flush:", err.Error())
			}

			fmt.Println("flush completed")

			idx, err := entity.NewIndexIvfFlat(entity.L2, 2)
			if err != nil {
				errChan <- fmt.Errorf("fail to create ivf flat index: %w", err)
			}
			if err = mvClient.CreateIndex(ctx, collectionName, "Vector", idx, false); err != nil {
				errChan <- fmt.Errorf("fail to create index: %w", err)
			}

			sidx := entity.NewScalarIndex()
			if err = mvClient.CreateIndex(ctx, collectionName, "ChunkID", sidx, false); err != nil {
				errChan <- fmt.Errorf("fail to create index: %w", err)
			}

			fmt.Println("index created")

			close(done)
		}()

		ctx := context.Background()
		has, err := mvClient.HasCollection(ctx, collectionName)
		if err != nil {
			errChan <- fmt.Errorf("fail to check collection: %w", err)
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

		// Step 2: insert all embedding data into the embeddings table in batches.
		const batchSize = 500
		chunkIDs := make([]int64, 0, batchSize)
		vectors := make([][]float32, 0, batchSize)

		updateChunksTable := func(chunkIDs []int64) {
			updateQuery := "UPDATE chunks SET processed = 1 WHERE id IN (" + placeholders(len(chunkIDs)) + ")"
			updateArgs := make([]any, len(chunkIDs))
			for i, chunkID := range chunkIDs {
				updateArgs[i] = chunkID
			}
			_, err := db.Exec(updateQuery, updateArgs...)
			if err != nil {
				errChan <- err
				return
			}

			fmt.Printf("Updated chunks table, batch size: %d\n", len(chunkIDs))
		}

		insertEmbeddings := func() {
			chunkIDColumn := entity.NewColumnInt64("ChunkID", chunkIDs)
			vectorColumn := entity.NewColumnFloatVector("Vector", 1536, vectors)
			_, err = mvClient.Insert(ctx, collectionName, "", chunkIDColumn, vectorColumn)
			if err != nil {
				errChan <- err
				return
			}

			fmt.Printf("Inserted into embeddings, batch size: %d\n", len(chunkIDs))

			updateChunksTable(chunkIDs)

			// Reset the slices for the next batch.
			chunkIDs = chunkIDs[:0]
			vectors = vectors[:0]
		}

		for emb := range embChan {
			chunkIDs = append(chunkIDs, int64(emb.chunkID))
			vectors = append(vectors, emb.data)

			if len(chunkIDs) >= batchSize {
				insertEmbeddings()
			}
		}

		if len(chunkIDs) > 0 {
			insertEmbeddings()
		}
	}()

	rows, err := db.Query(`SELECT id, content FROM chunks WHERE processed IS FALSE;`)
	if err != nil {
		return fmt.Errorf("fail to query db: %w", err)
	}
	defer rows.Close()

	maxWorkers := runtime.NumCPU()
	semaphore := make(chan struct{}, maxWorkers)
	// Step 1: calculate embeddings for all chunks in the DB, storing them in embs.
	for rows.Next() {
		var (
			id      int
			content string
		)
		if err = rows.Scan(&id, &content); err != nil {
			close(embChan)
			return fmt.Errorf("fail to scan row: %w", err)
		}

		wg.Add(1)
		semaphore <- struct{}{}
		go func(id int, content string) {
			defer wg.Done()
			defer func() { <-semaphore }()
			// Check for previous errors to avoid unnecessary work.
			select {
			case <-errChan:
				return
			default:
				fmt.Printf("id: %d, content size: %d\n", id, len(content))
				if len(content) > 0 {
					embeddings, err := getEmbeddings(client, content)
					if err != nil {
						errChan <- fmt.Errorf("fail to get embeddings: %w", err)
						return
					}

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
		}(id, content)
	}

	if err = rows.Err(); err != nil {
		return fmt.Errorf("fail to iterate rows: %w", err)
	}

	wg.Wait()
	close(embChan)

	select {
	case err := <-errChan:
		return err
	default:
		<-done // Wait for the insertion routine to finish
	}

	return nil
}

// Helper function to generate placeholders for the UPDATE query
func placeholders(n int) string {
	return strings.Repeat("?,", n-1) + "?"
}

// getEmbedding invokes the OpenAI embedding API to calculate the embedding
// for the given string. It returns the embedding.
func getEmbeddings(client *openai.Client, data string) ([][]float32, error) {
	var embeddings [][]float32
	const maxTokensPerRequest = 8192
	if len(data) <= maxTokensPerRequest {
		emb, err := getEmbedding(client, data)
		if err != nil {
			return nil, fmt.Errorf("fail to get embedding: %w", err)
		}

		embeddings = append(embeddings, emb)
		return embeddings, nil
	}

	chunks := splitIntoChunks(data, maxTokensPerRequest)
	for _, chunk := range chunks {
		emb, err := getEmbedding(client, chunk)
		if err != nil {
			return nil, fmt.Errorf("fail to get embedding: %w", err)
		}
		embeddings = append(embeddings, emb)
	}
	return embeddings, nil
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

func getEmbedding(client *openai.Client, data string) ([]float32, error) {
	queryReq := openai.EmbeddingRequest{
		Input: []string{data},
		Model: openai.SmallEmbedding3,
	}

	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		return nil, fmt.Errorf("fail to create embeddings: %w", err)
	}
	return queryResponse.Data[0].Embedding, nil
}

// encodeEmbedding encodes an embedding into a byte buffer, e.g. for DB storage as a blob.
func encodeEmbedding(emb []float32) ([]byte, error) {
	buf := make([]byte, len(emb)*4)
	if err := binary.Write(bytes.NewBuffer(buf), binary.LittleEndian, emb); err != nil {
		return nil, err
	}
	return buf, nil
}

// decodeEmbedding decodes an embedding back from a byte buffer.
func decodeEmbedding(b []byte) ([]float32, error) {
	// Calculate how many float32 values are in the slice.
	count := len(b) / 4
	numbers := make([]float32, count)
	if err := binary.Read(bytes.NewBuffer(b), binary.LittleEndian, numbers); err != nil {
		return nil, err
	}

	return numbers, nil
}
