package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8001"
	}

	// Serve static files from /static subfolder
	fs := http.FileServer(http.Dir("/static"))
	http.Handle("/", http.StripPrefix("/", fs))

	// Health check
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	log.Printf("Starting web server on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}