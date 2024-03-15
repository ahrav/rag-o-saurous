package main

import (
	"context"
	"database/sql"
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

// main function is the entry point of the program.
// It parses command line flags and initializes the necessary clients.
// Depending on the flags, it either calculates embeddings and updates the DB or answers a question.
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

// answerQuestion function gets the embedding for the question, searches for similar embeddings in the DB,
// retrieves the corresponding chunks, and generates an answer using either a local model or the OpenAI API.
func answerQuestion(ctx context.Context, client *openai.Client, mvClient mvclient.Client, question string, useLocal bool) error {
	qEmb, err := getEmbedding(client, question)
	if err != nil {
		return fmt.Errorf("fail to get question embedding: %w", err)
	}
	// Create a float vector from the question embedding.
	vector := entity.FloatVector(qEmb)

	// Create a search param for the HNSW index.
	sp, err := entity.NewIndexHNSWSearchParam(64)
	if err != nil {
		return fmt.Errorf("fail to create search param: %w", err)
	}

	// Search for similar embeddings concurrently in the DB. We search using two queries:
	// 1. Search for similar embeddings for Go files with a higher topK value.
	// 2. Search for similar embeddings for non-Go files with a lower topK value.
	var results []mvclient.SearchResult
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sr, err := mvClient.Search(ctx, collectionName, nil, `FileType == ".go"`, []string{"ChunkID"}, []entity.Vector{vector}, "Vector",
			entity.COSINE, 2, sp)
		if err != nil {
			panic(err)
		}
		results = append(results, sr...)
	}()

	sr, err := mvClient.Search(ctx, collectionName, nil, `FileType != ".go"`, []string{"ChunkID"}, []entity.Vector{vector}, "Vector",
		entity.COSINE, 1, sp)
	if err != nil {
		panic(err)
	}
	results = append(results, sr...)

	wg.Wait()

	// Retrieve the corresponding chunks for the similar embeddings.
	var contextInfo strings.Builder
	for _, result := range results {
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
			contextInfo.WriteString(content)
		}
	}

	// Build the prompt and execute the LLM API.
	query := fmt.Sprintf(`Use the below information to answer the subsequent question.
Information:
%v

Question: %v`, contextInfo.String(), question)

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
		chunkID  int
		fileType string
		data     []float32
	}

	// Prepare for concurrent embedding generation.
	embChan := make(chan embData, 100)
	errChan := make(chan error, 1)
	done := make(chan struct{})
	var wg sync.WaitGroup

	// Concurrently generate embeddings for all chunks in the DB, storing them in embs.
	// Then, insert all embedding data into the embeddings table in batches.
	// Finally, create indexes for the embeddings table.
	go func() {
		defer func() {
			ctx, cancel := context.WithTimeout(ctx, time.Second*30)
			defer cancel()
			if err = mvClient.Flush(ctx, collectionName, false); err != nil {
				log.Fatal("fail to flush:", err.Error())
			}

			fmt.Println("flush completed")

			idx, err := entity.NewIndexHNSW(entity.COSINE, 16, 64)
			if err != nil {
				errChan <- fmt.Errorf("fail to create HNSW index: %w", err)
			}
			if err = mvClient.CreateIndex(ctx, collectionName, "Vector", idx, false); err != nil {
				errChan <- fmt.Errorf("fail to create index: %w", err)
			}

			sidx := entity.NewScalarIndex()
			if err = mvClient.CreateIndex(ctx, collectionName, "ChunkID", sidx, false); err != nil {
				errChan <- fmt.Errorf("fail to create index for ChunkID: %w", err)
			}

			if err = mvClient.CreateIndex(ctx, collectionName, "FileType", sidx, false); err != nil {
				errChan <- fmt.Errorf("fail to create index for FileTyp: %w", err)
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
				WithField(entity.NewField().WithName("FileType").WithDataType(entity.FieldTypeVarChar).WithMaxLength(16)).
				WithField(entity.NewField().WithName("Vector").WithDataType(entity.FieldTypeFloatVector).WithDim(3072))

			if err = mvClient.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
				fmt.Printf("ERROR: %v\n", err)
				errChan <- err
				return
			}

			fmt.Println("Collection created")
		}

		// Step 2: insert all embedding data into the embeddings table in batches.
		const batchSize = 500
		chunkIDs := make([]int64, 0, batchSize)
		fileTypes := make([]string, 0, batchSize)
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
			vectorColumn := entity.NewColumnFloatVector("Vector", 3072, vectors)
			fileTypeColumn := entity.NewColumnVarChar("FileType", fileTypes)
			_, err = mvClient.Insert(ctx, collectionName, "", chunkIDColumn, fileTypeColumn, vectorColumn)
			if err != nil {
				errChan <- err
				return
			}

			fmt.Printf("Inserted into embeddings, batch size: %d\n", len(chunkIDs))

			updateChunksTable(chunkIDs)

			// Reset the slices for the next batch.
			chunkIDs = chunkIDs[:0]
			fileTypes = fileTypes[:0]
			vectors = vectors[:0]
		}

		for emb := range embChan {
			chunkIDs = append(chunkIDs, int64(emb.chunkID))
			vectors = append(vectors, emb.data)
			fileTypes = append(fileTypes, emb.fileType)

			if len(chunkIDs) >= batchSize {
				insertEmbeddings()
			}
		}

		if len(chunkIDs) > 0 {
			insertEmbeddings()
		}
	}()

	// rows, err := db.Query(`SELECT id, content FROM chunks WHERE processed IS FALSE;`)
	rows, err := db.Query(`
		SELECT c.id, c.content, ft.file_type
		FROM file_types ft
		INNER JOIN chunks c ON ft.id = c.file_type_id
		WHERE c.processed IS FALSE;
	`)
	if err != nil {
		return fmt.Errorf("fail to query db: %w", err)
	}
	defer rows.Close()

	maxWorkers := runtime.NumCPU()
	semaphore := make(chan struct{}, maxWorkers)
	// Step 1: calculate embeddings for all chunks in the DB, storing them in embs.
	for rows.Next() {
		type chunk struct {
			id       int
			content  string
			fileType string
		}
		var c chunk

		if err = rows.Scan(&c.id, &c.content, &c.fileType); err != nil {
			close(embChan)
			return fmt.Errorf("fail to scan row: %w", err)
		}

		wg.Add(1)
		semaphore <- struct{}{}
		go func(chunk chunk) {
			defer wg.Done()
			defer func() { <-semaphore }()
			// Check for previous errors to avoid unnecessary work.
			select {
			case <-errChan:
				return
			default:
				fmt.Printf("id: %d, content size: %d file type: %s\n", chunk.id, len(chunk.content), chunk.fileType)
				if len(chunk.content) > 0 {
					embeddings, err := getEmbeddings(client, chunk.content)
					if err != nil {
						errChan <- fmt.Errorf("fail to get embeddings: %w", err)
						return
					}

					for _, emb := range embeddings {
						select {
						case embChan <- embData{chunkID: chunk.id, data: emb, fileType: chunk.fileType}:
						case err := <-errChan:
							log.Println("Error received, stopping:", err)
							return
						}
					}
				}
			}
		}(c)
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
// for the given string. if the string is too long, it splits it into chunks and
// calculates the embedding for each chunk. It returns the embeddings.
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
			// Only add a space if the builder is not empty.
			builder.WriteRune(' ')
		}
		builder.WriteString(word)
	}

	// Add the last chunk if it's not empty.
	if builder.Len() > 0 {
		chunks = append(chunks, builder.String())
	}

	return chunks
}

// getEmbedding function invokes the OpenAI embedding API to calculate the embedding
// for the given string. It returns the embedding.
func getEmbedding(client *openai.Client, data string) ([]float32, error) {
	queryReq := openai.EmbeddingRequest{
		Input: []string{data},
		Model: openai.LargeEmbedding3,
	}

	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		return nil, fmt.Errorf("fail to create embeddings: %w", err)
	}
	return queryResponse.Data[0].Embedding, nil
}
