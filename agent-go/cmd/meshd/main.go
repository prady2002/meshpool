package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"meshd/internal/agent"
	"meshd/internal/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	ag, err := agent.New(cfg)
	if err != nil {
		log.Fatalf("create agent: %v", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := ag.Run(ctx); err != nil {
		log.Fatalf("run agent: %v", err)
	}
}
