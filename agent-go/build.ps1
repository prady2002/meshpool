$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    go build ./cmd/meshd
    $env:CGO_ENABLED = '0'
    $env:GOOS = 'linux'
    $env:GOARCH = 'amd64'
    go build -o meshd-linux-amd64 ./cmd/meshd
}
finally {
    Pop-Location
}
