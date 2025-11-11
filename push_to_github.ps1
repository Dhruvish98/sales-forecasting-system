# PowerShell script to push to GitHub
# Make sure you've created the repository on GitHub first

Write-Host "Setting up GitHub remote..." -ForegroundColor Green

# Repository name (change if you used a different name)
$repoName = "sales-forecasting-system"
$username = "Dhruvish98"

# Add remote
Write-Host "Adding remote origin..." -ForegroundColor Yellow
git remote add origin "https://github.com/$username/$repoName.git"

# Check if remote was added
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote added successfully!" -ForegroundColor Green
} else {
    Write-Host "Remote might already exist. Checking..." -ForegroundColor Yellow
    git remote set-url origin "https://github.com/$username/$repoName.git"
}

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: https://github.com/$username/$repoName" -ForegroundColor Cyan
} else {
    Write-Host "Push failed. Make sure:" -ForegroundColor Red
    Write-Host "1. Repository exists on GitHub" -ForegroundColor Red
    Write-Host "2. You have proper authentication set up" -ForegroundColor Red
    Write-Host "3. Repository name matches: $repoName" -ForegroundColor Red
}

