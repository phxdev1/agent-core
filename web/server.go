package main

import (
	"embed"
	"flag"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

//go:embed *.html *.css *.js
var staticFiles embed.FS

func main() {
	port := flag.String("port", "8001", "Port to serve on")
	flag.Parse()

	if envPort := os.Getenv("PORT"); envPort != "" {
		*port = envPort
	}

	// Create file server from embedded files
	fsys, err := fs.Sub(staticFiles, ".")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()

	// Serve static files with caching headers
	mux.Handle("/", addHeaders(http.FileServer(http.FS(fsys))))

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Listen on all interfaces (0.0.0.0 binds to both IPv4 and IPv6)
	server := &http.Server{
		Addr:           "0.0.0.0:" + *port,
		Handler:        mux,
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1MB
	}

	log.Printf("Starting web server on port %s", *port)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func addHeaders(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Security headers
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "SAMEORIGIN")
		w.Header().Set("X-XSS-Protection", "1; mode=block")

		// Cache static assets
		if strings.HasSuffix(r.URL.Path, ".css") || strings.HasSuffix(r.URL.Path, ".js") {
			w.Header().Set("Cache-Control", "public, max-age=31536000")
		}

		// Serve index.html for root
		if r.URL.Path == "/" {
			r.URL.Path = "/index.html"
		}

		h.ServeHTTP(w, r)
	})
}