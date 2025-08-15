package main

import (
	"net/http"
	"os"
	"time"
)

func main() {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}
	
	port := "8001"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}
	
	resp, err := client.Get("http://localhost:" + port + "/health")
	if err != nil {
		os.Exit(1)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		os.Exit(1)
	}
	
	os.Exit(0)
}