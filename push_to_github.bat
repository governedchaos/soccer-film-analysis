@echo off
REM ============================================================
REM Soccer Film Analysis - Push to GitHub Script (Windows)
REM ============================================================
REM Usage: push_to_github.bat YOUR_GITHUB_USERNAME
REM ============================================================

setlocal enabledelayedexpansion

set REPO_NAME=soccer_film_analysis

echo.
echo ============================================================
echo    PUSH TO GITHUB
echo ============================================================
echo.

REM Check if username provided
if "%~1"=="" (
    echo Usage: push_to_github.bat YOUR_GITHUB_USERNAME
    echo.
    echo Example: push_to_github.bat johnsmith
    echo.
    echo Make sure you've created the repository on GitHub first:
    echo   https://github.com/new
    echo   Repository name: %REPO_NAME%
    echo   Do NOT initialize with README
    pause
    exit /b 1
)

set GITHUB_USER=%~1
set REMOTE_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo GitHub Username: %GITHUB_USER%
echo Repository: %REPO_NAME%
echo Remote URL: %REMOTE_URL%
echo.

REM Check if git is initialized
if not exist ".git" (
    echo [1/4] Initializing git repository...
    git init
) else (
    echo [1/4] Git repository already initialized
)

REM Add all files
echo [2/4] Staging files...
git add .

REM Commit
echo [3/4] Creating commit...
git commit -m "Initial commit: Soccer film analysis application"

REM Set up remote and push
echo [4/4] Pushing to GitHub...

git remote remove origin 2>nul
git remote add origin %REMOTE_URL%
git branch -M main
git push -u origin main

echo.
echo ============================================================
echo    PUSHED TO GITHUB
echo ============================================================
echo.
echo Repository URL: https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
echo Next: Use Claude Code to test and improve the application
echo.
pause
