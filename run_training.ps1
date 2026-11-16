param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [string]$Output = "artifacts",
    [int]$Epochs = 10,
    [int]$BatchSize = 128,
    [switch]$AllowDifferentClassCount
)

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $uv) {
    $uvPath = $uv.Source
} else {
    $uvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uvPath)) {
        throw "uv was not found. Install uv or add it to PATH."
    }
}

$arguments = @(
    (Join-Path $PSScriptRoot "train_character_recognizer.py"),
    "--dataset", $Dataset,
    "--output", $Output,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize
)
if ($AllowDifferentClassCount) {
    $arguments += "--allow-different-class-count"
}

& $uvPath run --python 3.11 --with-requirements (Join-Path $PSScriptRoot "requirements.txt") @arguments
exit $LASTEXITCODE
