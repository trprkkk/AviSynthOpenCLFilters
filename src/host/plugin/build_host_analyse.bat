@echo off
rem build_host_analyse.bat — plain cl.exe build of the NINTH host plugin
rem (KMAnalyseOCL AviSynth+ DLL) + its headless verification exe.  Run from
rem repo\src\host\plugin.
rem
rem This plugin embeds TWO .cl units (ktgmc_motion.cl: prepare/global-MV/interp/
rem calc_all_sad kernels; ktgmc_search.cl: the ordered-column block-search
rem kernel), so the embed step runs twice.  NOTE: build_host_analyze.bat (no
rem "se") is the SEVENTH plugin (KFM analyze) — different file.
rem
rem Prereqs (same verified MSVC x64 recipe as the other plugin builds):
rem   MSVC 14.51 (VS 18 Community), Windows Kits 10.0.26100.0,
rem   OpenCL headers at Q:\Downloads\openclfilter\sdk\opencl\include,
rem   OpenCL.lib at Q:\Downloads\openclfilter\sdk\opencl\lib\x64,
rem   AviSynth+ SDK headers at Q:\Downloads\openclfilter\sdk.
rem
rem The chained KMSuperOCL plugin (ktgmc_super_avs.cpp, eighth plugin) is
rem rebuilt too: KMAnalyseOCL consumes the SUPER_FRAME KMVParam it smuggles
rem into the VideoInfo (ktgmc_kmparam.h), and the verifier installs both DLLs.

setlocal enabledelayedexpansion

set MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231
set WINKITS=C:\Program Files (x86)\Windows Kits\10
set WINVER=10.0.26100.0
set ROOT=%~dp0..\..\..
set SDKROOT=Q:\Downloads\openclfilter\sdk
set OUTDIR=%~dp0build_host

set "PATH=%MSVC%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%WINKITS%\Include\%WINVER%\shared;%WINKITS%\Include\%WINVER%\ucrt;%WINKITS%\Include\%WINVER%\um;%WINKITS%\Include\%WINVER%\winrt;%SDKROOT%\opencl\include"
set "LIB=%MSVC%\lib\x64;%WINKITS%\Lib\%WINVER%\ucrt\x64;%WINKITS%\Lib\%WINVER%\um\x64;%SDKROOT%\opencl\lib\x64"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

cd /d "%~dp0"

rem 1. embed the two kernel sources into the DLL
python3 "%~dp0embed_cl.py" "%ROOT%\src\opencl\ktgmc\kernels\ktgmc_motion.cl" "%OUTDIR%\ktgmc_motion_cl.h" ktgmc_motion_cl_src
if errorlevel 1 exit /b 1
python3 "%~dp0embed_cl.py" "%ROOT%\src\opencl\ktgmc\kernels\ktgmc_search.cl" "%OUTDIR%\ktgmc_search_cl.h" ktgmc_search_cl_src
if errorlevel 1 exit /b 1
copy /y "%OUTDIR%\ktgmc_motion_cl.h" "%~dp0ktgmc_motion_cl.h" >nul
copy /y "%OUTDIR%\ktgmc_search_cl.h" "%~dp0ktgmc_search_cl.h" >nul

rem 2. the KMAnalyseOCL plugin DLL
cl /nologo /EHsc /MD /std:c++17 /W3 /LD ^
   ktgmc_analyse_avs.cpp "%ROOT%\src\host\ocl_host.cpp" ^
   /I"%ROOT%\src\host" /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\ktgmc_analyse_avs.dll" ^
   /link OpenCL.lib /OUT:"%OUTDIR%\ktgmc_analyse_avs.dll"
if errorlevel 1 exit /b 1

rem 3. headless verification exe
cl /nologo /EHsc /MD /std:c++17 /W3 ^
   test_ktgmc_analyse.cpp ^
   /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\test_ktgmc_analyse.exe" ^
   /link /OUT:"%OUTDIR%\test_ktgmc_analyse.exe"
if errorlevel 1 exit /b 1

echo.
echo build OK:
dir /b "%OUTDIR%\ktgmc_analyse_avs.dll" "%OUTDIR%\test_ktgmc_analyse.exe"
endlocal
