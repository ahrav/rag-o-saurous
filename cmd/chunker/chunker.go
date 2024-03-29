package main

import (
	"bufio"
	"bytes"
	"database/sql"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkoukk/tiktoken-go"
	"golang.org/x/net/html"
)

const (
	tableSql = `
CREATE TABLE IF NOT EXISTS %v (
  id INTEGER PRIMARY KEY,
  file_type TEXT UNIQUE
);

INSERT OR IGNORE INTO %v (id, file_type) VALUES
  (1, '.go'),
  (2, '.html'),
  (3, '.md');

CREATE TABLE IF NOT EXISTS %v (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT,
  nchunk INTEGER,
  content TEXT,
  processed INTEGER DEFAULT 0,
  file_type_id INTEGER,
  FOREIGN KEY (file_type_id) REFERENCES %v (id)
);

CREATE INDEX IF NOT EXISTS idx_processed ON %v (processed);
CREATE INDEX IF NOT EXISTS idx_file_type_id ON %v (file_type_id);
`
	tableName         = "chunks"
	fileTypeTableName = "file_types"

	chunkSize = 512
)

func main() {
	rootDir := flag.String("rootdir", ".", "root directory to take source code files from")
	doClear := flag.Bool("clear", false, "clear DB table before inserting")
	outDb := flag.String("outdb", "chunks.db", "filename for sqlite DB for output")
	dryRun := flag.Bool("dryrun", false, "do not write to DB")
	fileExtensions := flag.String("file-extensions", ".go", "comma separated list of file extensions to chunk")
	flag.Parse()

	exts := strings.Split(*fileExtensions, ",")
	extensions := make(map[string]string, len(exts))
	for _, ext := range exts {
		s, ok := strings.CutPrefix(ext, ".")
		if !ok {
			log.Fatalf("invalid extension %v", ext)
		}
		extensions[ext] = s
	}

	db, err := sql.Open("sqlite3", *outDb)
	checkErr(err)

	stmt := fmt.Sprintf(tableSql, fileTypeTableName, fileTypeTableName, tableName, fileTypeTableName, tableName, tableName)
	_, err = db.Exec(stmt)
	checkErr(err)

	if *doClear {
		fmt.Printf("Clearing DB table %v", tableName)
		_, err := db.Exec(fmt.Sprintf("delete from %s", tableName))
		checkErr(err)
	}

	insertStmt, err := db.Prepare("insert into chunks(path, nchunk, content, processed, file_type_id) values(?, ?, ?, 0, ?)")
	checkErr(err)

	// Initialize the token encoder.
	tokenEncoder, err := tiktoken.GetEncoding(tiktoken.MODEL_CL100K_BASE)
	checkErr(err)

	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	semaphore := make(chan struct{}, numCPU)

	const batchSize = 100
	var batchesMutex sync.Mutex
	var batches [][]any

	var tokTotal atomic.Uint64
	err = filepath.WalkDir(*rootDir, func(path string, d fs.DirEntry, err error) error {
		if val, exists := extensions[filepath.Ext(path)]; exists {
			wg.Add(1)
			semaphore <- struct{}{}

			go func(path string, val string) {
				defer func() {
					<-semaphore
					wg.Done()
				}()

				log.Printf("Chunking %v", path)

				var chunks []string
				var fileTypeID int
				if val == "go" {
					if strings.HasSuffix(path, "_test.go") || strings.HasSuffix(path, ".pb.go") || strings.HasSuffix(path, ".pb.validate.go") {
						return
					}
					chunks, err = chunkGoFile(tokenEncoder, path)
					fileTypeID = 1
				} else if val == "html" {
					chunks, err = chunkHtmlFile(tokenEncoder, path)
					fileTypeID = 2
				} else {
					chunks, err = chunkTextFile(tokenEncoder, path)
					fileTypeID = 3
				}
				if err != nil {
					log.Printf("Error chunking %v: %v", path, err)
					return
				}

				var batch []any
				for i, chunk := range chunks {
					fmt.Println(path, i, len(chunk))
					tokTotal.Add(uint64(len(chunk)))
					if *dryRun {
						continue
					}
					batch = append(batch, path, i, chunk, fileTypeID)

					if len(batch)/4 >= batchSize {
						batchesMutex.Lock()
						batches = append(batches, batch)
						if len(batches) >= batchSize {
							for _, batch := range batches {
								for i := 0; i < len(batch); i += 4 {
									_, err := insertStmt.Exec(batch[i], batch[i+1], batch[i+2], batch[i+3])
									checkErr(err)
								}
							}
							batches = nil
						}
						batchesMutex.Unlock()
						batch = nil
					}

				}

				if len(batch) > 0 {
					batchesMutex.Lock()
					batches = append(batches, batch)
					batchesMutex.Unlock()
				}

			}(path, val)
		}

		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	wg.Wait()

	if len(batches) > 0 {
		for _, batch := range batches {
			for i := 0; i < len(batch); i += 4 {
				_, err := insertStmt.Exec(batch[i], batch[i+1], batch[i+2], batch[i+3])
				if err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	fmt.Println("Total tokens:", tokTotal.Load())
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

// chunkGoFile reads the Go file in `path` and breaks it into chunks of
// approximately chunkSize tokens each, returning the chunks.
// It leverages the go parser to extract the code snippets.
func chunkGoFile(encoder *tiktoken.Tiktoken, path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	src, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		return nil, err
	}

	var chunks []string
	var currentChunk string

	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		// TODO: Do we need other types of nodes?
		case *ast.FuncDecl, *ast.GenDecl, *ast.TypeSpec, *ast.InterfaceType, *ast.StructType, *ast.File:
			// Extract the code snippet for functions, methods, structs, imports, constants, and variables.
			start := fset.Position(x.Pos()).Offset
			end := fset.Position(x.End()).Offset
			snippet := string(src[start:end])

			// Append the snippet to the current chunk.
			currentChunk += snippet + "\n"

			// Encode the current chunk and check if it exceeds the chunk size.
			toks := encoder.Encode(currentChunk, nil, nil)
			if len(toks) > chunkSize {
				// If the chunk exceeds the size, add it to the chunks slice and start a new chunk.
				chunks = append(chunks, currentChunk)
				currentChunk = ""
			}
		}
		return true
	})

	// Append the remaining code snippet to the current chunk.
	chunks = append(chunks, currentChunk)

	return chunks, nil
}

func chunkTextFile(encoder *tiktoken.Tiktoken, path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(scanParagraphs)

	var chunks []string
	var currentChunk string

	for scanner.Scan() {
		// Append the snippet to the current chunk.
		currentChunk += scanner.Text()

		// Encode the current chunk and check if it exceeds the chunk size.
		toks := encoder.Encode(currentChunk, nil, nil)
		if len(toks) > chunkSize {
			// If the chunk exceeds the size, add it to the chunks slice and start a new chunk.
			chunks = append(chunks, currentChunk)
			currentChunk = ""
		}
	}

	// Append the remaining code snippet to the current chunk.
	chunks = append(chunks, currentChunk)

	return chunks, nil
}

func scanParagraphs(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}

	if i := bytes.Index(data, []byte("\n\n")); i >= 0 {
		return i + 2, data[0:i], nil
	}

	if atEOF {
		return len(data), data, nil
	}

	return 0, nil, nil
}

func chunkHtmlFile(encoder *tiktoken.Tiktoken, path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	doc, err := html.Parse(f)
	if err != nil {
		return nil, err
	}

	var chunks []string
	var currentChunk string

	var extractText func(*html.Node)
	extractText = func(n *html.Node) {
		if n.Type == html.ElementNode && (n.Data == "script" || n.Data == "style") {
			return // Skip script and style elements
		}
		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				currentChunk += text + " "

				toks := encoder.Encode(currentChunk, nil, nil)
				if len(toks) > chunkSize {
					chunks = append(chunks, strings.TrimSpace(currentChunk))
					currentChunk = ""
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			extractText(c)
		}
	}

	var skipElement func(*html.Node)
	skipElement = func(n *html.Node) {
		if n.Type == html.ElementNode && (n.Data == "script" || n.Data == "style") {
			return
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			skipElement(c)
		}
	}

	skipElement(doc)
	extractText(doc)

	if currentChunk != "" {
		chunks = append(chunks, strings.TrimSpace(currentChunk))
	}

	return chunks, nil
}
