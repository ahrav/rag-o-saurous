package main

import (
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
	"strings"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkoukk/tiktoken-go"
)

const tokenEncoding = "cl100k_base"
const tableName = "chunks"
const chunkSize = 1000

const tableSql = `CREATE TABLE IF NOT EXISTS %v (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT,
  nchunk INTEGER,
  content TEXT
);`

func main() {
	rootDir := flag.String("rootdir", ".", "root directory to take source code files from")
	doClear := flag.Bool("clear", false, "clear DB table before inserting")
	outDb := flag.String("outdb", "chunks.db", "filename for sqlite DB for output")
	dryRun := flag.Bool("dryrun", false, "do not write to DB")
	flag.Parse()

	db, err := sql.Open("sqlite3", *outDb)
	checkErr(err)

	_, err = db.Exec(fmt.Sprintf(tableSql, tableName))
	checkErr(err)

	if *doClear {
		log.Printf("Clearing DB table %v", tableName)
		_, err := db.Exec(fmt.Sprintf("delete from %s", tableName))
		checkErr(err)
	}

	insertStmt, err := db.Prepare("insert into chunks(path, nchunk, content) values(?, ?, ?)")
	checkErr(err)

	// Initialize the token encoder.
	tokenEncoder, err := tiktoken.GetEncoding(tokenEncoding)
	checkErr(err)

	tokTotal := 0
	err = filepath.WalkDir(*rootDir, func(path string, d fs.DirEntry, err error) error {
		if filepath.Ext(path) == ".go" && !strings.Contains(path, "_test.go") {
			log.Printf("Chunking %v", path)
			chunks := chunkFile(tokenEncoder, path)

			for i, chunk := range chunks {
				fmt.Println(path, i, len(chunk))
				tokTotal += len(chunk)
				if *dryRun {
					continue
				}
				_, err := insertStmt.Exec(path, i, chunk)
				checkErr(err)
			}

		}
		return nil
	})
	fmt.Println("Total tokens:", tokTotal)
}

func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

// chunkFile reads the file in `path` and breaks it into chunks of
// approximately chunkSize tokens each, returning the chunks.
func chunkFile(encoder *tiktoken.Tiktoken, path string) []string {
	f, err := os.Open(path)
	checkErr(err)
	defer f.Close()

	src, err := io.ReadAll(f)
	checkErr(err)

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	checkErr(err)

	var chunks []string
	var currentChunk string

	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		// TODO: Do we need other types of nodes?
		case *ast.FuncDecl, *ast.GenDecl, *ast.TypeSpec, *ast.InterfaceType:
			// Extract the code snippet for functions, methods, structs, imports, constants, and variables.
			start := fset.Position(x.Pos()).Offset
			end := fset.Position(x.End()).Offset
			snippet := string(src[start:end])

			// Append the snippet to the current chunk.
			currentChunk += snippet

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

	return chunks
}
