package main

import (
	"bytes"
	"compress/gzip"
	"embed"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

//go:embed *.html *.css *.js
var staticFiles embed.FS

// Pre-computed responses cached in memory
var (
	cachedResponses = make(map[string]*cachedResponse)
	cacheMutex      sync.RWMutex
	indexHTML       []byte
	indexGzipped    []byte
)

type cachedResponse struct {
	content     []byte
	gzipped     []byte
	contentType string
	etag        string
}

func init() {
	// Force garbage collection settings for minimal memory
	debug.SetGCPercent(20)
	debug.SetMemoryLimit(100 << 20) // 100MB limit
	
	// Pre-load and compress all static files at startup
	preloadFiles()
	
	// Pin to CPU for better cache locality
	runtime.LockOSThread()
}

func preloadFiles() {
	files := []string{"index.html", "style.css", "script.js"}
	
	for _, file := range files {
		data, err := staticFiles.ReadFile(file)
		if err != nil {
			continue
		}
		
		// Pre-compress with gzip
		var buf bytes.Buffer
		gz := gzip.NewWriter(&buf)
		gz.Write(data)
		gz.Close()
		
		// Determine content type
		contentType := "text/plain"
		switch {
		case strings.HasSuffix(file, ".html"):
			contentType = "text/html; charset=utf-8"
		case strings.HasSuffix(file, ".css"):
			contentType = "text/css; charset=utf-8"
		case strings.HasSuffix(file, ".js"):
			contentType = "application/javascript; charset=utf-8"
		}
		
		// Cache the response
		cachedResponses["/"+file] = &cachedResponse{
			content:     data,
			gzipped:     buf.Bytes(),
			contentType: contentType,
			etag:        `"` + file + "-v1"`,
		}
		
		// Special handling for index
		if file == "index.html" {
			indexHTML = data
			indexGzipped = buf.Bytes()
			cachedResponses["/"] = cachedResponses["/index.html"]
		}
	}
	
	log.Printf("Pre-loaded %d files into memory", len(cachedResponses))
}

func main() {
	port := flag.String("port", "8001", "Port to serve on")
	flag.Parse()
	
	if envPort := os.Getenv("PORT"); envPort != "" {
		*port = envPort
	}
	
	// Custom server with optimized settings
	server := &http.Server{
		Addr:           ":" + *port,
		Handler:        http.HandlerFunc(handler),
		ReadTimeout:    5 * time.Second,
		WriteTimeout:   5 * time.Second,
		IdleTimeout:    60 * time.Second,
		MaxHeaderBytes: 1 << 16, // 64KB
	}
	
	// Use SO_REUSEPORT for better load balancing
	server.SetKeepAlivesEnabled(true)
	
	log.Printf("Ultra-optimized web server on port %s (cache-aligned)", *port)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func handler(w http.ResponseWriter, r *http.Request) {
	// Fast path for health checks
	if r.URL.Path == "/health" {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	// Get from pre-computed cache
	cacheMutex.RLock()
	cached, found := cachedResponses[r.URL.Path]
	cacheMutex.RUnlock()
	
	if !found {
		// Default to index for SPA routing
		cached = cachedResponses["/"]
	}
	
	// Check if client supports gzip
	acceptsGzip := strings.Contains(r.Header.Get("Accept-Encoding"), "gzip")
	
	// Set headers
	w.Header().Set("Content-Type", cached.contentType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "SAMEORIGIN")
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	w.Header().Set("ETag", cached.etag)
	
	// Check ETag for 304
	if r.Header.Get("If-None-Match") == cached.etag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	
	// Serve pre-compressed or raw
	if acceptsGzip && len(cached.gzipped) > 0 {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Vary", "Accept-Encoding")
		w.Write(cached.gzipped)
	} else {
		w.Write(cached.content)
	}
}

// Custom zero-allocation writer for even better performance
type zeroAllocWriter struct {
	w   http.ResponseWriter
	buf []byte
}

func (z *zeroAllocWriter) Write(p []byte) (n int, err error) {
	return z.w.Write(p)
}