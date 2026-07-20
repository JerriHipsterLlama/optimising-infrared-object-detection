param(
    [Parameter(Mandatory = $true)][string]$Engine,
    [int]$Iterations = 100,
    [int]$Warmup = 20
)

if (-not (Get-Command trtexec -ErrorAction SilentlyContinue)) {
    throw "trtexec was not found. Run this script on the Jetson with TensorRT installed."
}

trtexec --loadEngine=$Engine --iterations=$Iterations --warmUp=$Warmup --noDataTransfers --verbose

