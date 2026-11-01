@echo off
rem build_host_compensate.bat — plain cl.exe build of the TENTH host plugin
rem (KMCompensateOCL / KMDegrain1OCL / KMDegrain2OCL AviSynth+ DLLs) + its
rem headless verification exe.  Run from repo\src\host\plugin.
rem
rem This plugin embeds ONE .cl unit (ktgmc_motion.cl: load_mv_batch,
rem prepare_degrain/compensate, degrain_patch, overlap_out, scene_change
rem kernels).
rem
rem Prereqs (same verified MSVC x64 recipe as the other plugin builds):
rem   MSVC 14.51 (VS 18 Community), Windows Kits 10.0.26100.0,
rem   OpenCL headers at Q:\Downloads\openclfilter\sdk\opencl\include,
rem   OpenCL.lib at Q:\Downloads\openclfilter\sdk\opencl\lib\x64,
rem   AviSynth+ SDK headers at Q:\Downloads\openclfilter\sdk.
rem
rem The chained KMSuperOCL (eighth) and KMAnalyseOCL (ninth) plugins are
rem rebuilt too: KMCompensateOCL consumes the SUPER_FRAME and MV_FRAME
rem KMVParams they smuggle into the VideoInfo (ktgmc_kmparam.h).

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

rem 1. embed the kernel source into the DLL
python3 "%~dp0embed_cl.py" "%ROOT%\src\opencl\ktgmc\kernels\ktgmc_motion.cl" "%OUTDIR%\ktgmc_motion_cl.h" ktgmc_motion_cl_src
if errorlevel 1 exit /b 1
copy /y "%OUTDIR%\ktgmc_motion_cl.h" "%~dp0ktgmc_motion_cl.h" >nul

rem 2. rebuild the super plugin (KMAnalyseOCL consumes its SUPER_FRAME)
cl /nologo /EHsc /MD /std:c++17 /W3 /LD ^
   ktgmc_super_avs.cpp "%ROOT%\src\host\ocl_host.cpp" ^
   /I"%ROOT%\src\host" /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\ktgmc_super_avs.dll" ^
   /link OpenCL.lib /OUT:"%OUTDIR%\ktgmc_super_avs.dll"
if errorlevel 1 exit /b 1

rem 3. rebuild the analyse plugin (KMCompensateOCL consumes its MV_FRAME)
cl /nologo /EHsc /MD /std:c++17 /W3 /LD ^
   ktgmc_analyse_avs.cpp "%ROOT%\src\host\ocl_host.cpp" ^
   /I"%ROOT%\src\host" /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\ktgmc_analyse_avs.dll" ^
   /link OpenCL.lib /OUT:"%OUTDIR%\ktgmc_analyse_avs.dll"
if errorlevel 1 exit /b 1

rem 4. the compensate/degrain plugin DLL
cl /nologo /EHsc /MD /std:c++17 /W3 /LD ^
   ktgmc_compensate_avs.cpp "%ROOT%\src\host\ocl_host.cpp" ^
   /I"%ROOT%\src\host" /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\ktgmc_compensate_avs.dll" ^
   /link OpenCL.lib /OUT:"%OUTDIR%\ktgmc_compensate_avs.dll"
if errorlevel 1 exit /b 1

rem 5. headless verification exe
cl /nologo /EHsc /MD /std:c++17 /W3 ^
   test_ktgmc_compensate.cpp ^
   /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\test_ktgmc_compensate.exe" ^
   /link /OUT:"%OUTDIR%\test_ktgmc_compensate.exe"
if errorlevel 1 exit /b 1

echo.
echo build OK:
dir /b "%OUTDIR%\ktgmc_compensate_avs.dll" "%OUTDIR%\test_ktgmc_compensate.exe"
endlocal
