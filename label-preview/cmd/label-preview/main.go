package main

import (
	"os"

	"github.com/rellimrebo/toolbox/label-preview/internal/app"
)

func main() {
	os.Exit(app.Run(os.Args[1:], os.Stdout, os.Stderr))
}
