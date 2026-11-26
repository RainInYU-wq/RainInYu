@echo off
setlocal
cd /d "%~dp0"
where xmake >nul 2>nul || (echo Missing xmake. See README_CN.txt. & pause & exit /b 1)
where clang-cl >nul 2>nul || (echo Missing clang-cl. See README_CN.txt. & pause & exit /b 1)
call :build > build.log 2>&1
if errorlevel 1 (echo Build failed. Send build.log. & pause & exit /b 1)
echo Build succeeded. Look in bin\BEItemTransformInspector\
pause
exit /b 0
:build
xmake repo -u || exit /b 1
xmake f -c -p windows -a x64 -m release -y || exit /b 1
xmake -v -y || exit /b 1
if not exist "bin\BEItemTransformInspector\BEItemTransformInspector.dll" exit /b 1
if not exist "bin\BEItemTransformInspector\manifest.json" exit /b 1
exit /b 0
