param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [string]$Output = "artifacts",
    [int]$Epochs = 10,
    [int]$BatchSize = 128,
    [switch]$AllowDifferentClassCount
)

$python = "C:\tfenv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "TensorFlow environment not found at $python. Create it using the README instructions."
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

& $python @arguments
exit $LASTEXITCODE
