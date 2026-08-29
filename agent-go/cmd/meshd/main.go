package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"meshd/internal/agent"
	"meshd/internal/config"
	"meshd/internal/distribution"
	"meshd/internal/update"
)

var commit = "dev"

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "--version", "version":
			fmt.Printf("meshd %s (%s)\n", update.Version, commit)
			return
		case "service":
			if err := serviceCommand(os.Args[2:]); err != nil {
				log.Fatalf("service: %v", err)
			}
			return
		}
	}
	if cfgPath := configFlag(os.Args[1:]); cfgPath != "" {
		if err := config.LoadEnvFile(cfgPath); err != nil {
			log.Fatalf("load config file: %v", err)
		}
	} else if cfgPath := os.Getenv("MESHPOOL_CONFIG_FILE"); cfgPath != "" {
		if err := config.LoadEnvFile(cfgPath); err != nil {
			log.Fatalf("load config file: %v", err)
		}
	}
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

func configFlag(args []string) string {
	fs := flag.NewFlagSet("meshd", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	path := fs.String("config", "", "path to meshd env config")
	if err := fs.Parse(args); err != nil {
		return ""
	}
	return *path
}

func serviceCommand(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: meshd service install|uninstall|status")
	}
	action := args[0]
	fs := flag.NewFlagSet("meshd service", flag.ContinueOnError)
	server := fs.String("server", "", "control-plane URL")
	token := fs.String("token", "", "invite token")
	name := fs.String("name", "", "node name")
	cpu := fs.Int("cpu", 50, "donated CPU percentage")
	binary := fs.String("binary", "", "source meshd binary to install")
	root := fs.String("root", "", "installation root override")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	plan := distribution.PlanForCurrentOS(distribution.WithRoot(*root))
	switch action {
	case "install":
		if *server == "" {
			return fmt.Errorf("--server is required")
		}
		if *binary == "" {
			exe, err := os.Executable()
			if err != nil {
				return err
			}
			*binary = exe
		}
		if *name == "" {
			host, _ := os.Hostname()
			*name = host
		}
		return distribution.Install(distribution.InstallRequest{
			Plan:            plan,
			SourceBinary:    *binary,
			ControlPlaneURL: *server,
			EnrollToken:     *token,
			NodeName:        *name,
			DonateCPUPct:    *cpu,
		})
	case "uninstall":
		return distribution.Uninstall(distribution.UninstallRequest{Plan: plan})
	case "status":
		st, ok, err := distribution.SystemService(plan.Platform).Status(distribution.ServiceName)
		if err != nil {
			return err
		}
		if !ok {
			fmt.Println("not installed")
			return nil
		}
		if st.Started {
			fmt.Println("running")
		} else {
			fmt.Println("stopped")
		}
		return nil
	default:
		return fmt.Errorf("unknown service action %q", action)
	}
}
