@echo off
echo ============================================
echo   NapsGen - Roblox Image Scraper
echo ============================================
echo.

set /p COUNT="How many images to scrape? (default: 10000): "
if "%COUNT%"=="" set COUNT=10000

echo Scraping %COUNT% Roblox images to data/images/...
python -m data.scraper --output data/images --count %COUNT% --type all

echo.
echo Done! Now preprocess the images:
echo   python -m dataset.validate --data-dir data/images --resolution 64
echo   python -m dataset.quality_filter --data-dir data/images --output-dir dataset
echo.
pause
