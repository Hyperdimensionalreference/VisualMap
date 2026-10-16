Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"

!ifndef BUNDLE
  !error "BUNDLE must point at the verified standalone application directory"
!endif
!ifndef OUTPUT
  !error "OUTPUT must specify the installer EXE path"
!endif
!ifndef DELETE_MANIFEST
  !error "DELETE_MANIFEST must list only files owned by this application"
!endif

Name "VisualMap"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\VisualMap"
InstallDirRegKey HKCU "Software\VisualMap" "InstallDir"
RequestExecutionLevel user
AllowRootDirInstall false
SetCompressor /SOLID lzma
SetCompressorDictSize 32
CRCCheck on
ShowInstDetails show
ShowUninstDetails show
VIProductVersion "0.1.0.0"
VIAddVersionKey /LANG=2052 "ProductName" "VisualMap"
VIAddVersionKey /LANG=2052 "CompanyName" "Hyperdimensionalreference"
VIAddVersionKey /LANG=2052 "FileDescription" "VisualMap 本机视频与空间工作台安装程序"
VIAddVersionKey /LANG=2052 "FileVersion" "0.1.0"
VIAddVersionKey /LANG=2052 "LegalCopyright" "Copyright (c) 2026 VisualMap Team; third-party licenses apply."

!define MUI_ABORTWARNING
!define MUI_ICON "${BUNDLE}\VisualMap.ico"
!define MUI_UNICON "${BUNDLE}\VisualMap.ico"
!define MUI_WELCOMEPAGE_TITLE "安装 VisualMap"
!define MUI_WELCOMEPAGE_TEXT "本机视频库、离线识别与三维空间工作台。$\r$\n$\r$\n安装包已含运行环境和所需模型，无需另装 Python。个人视频和结果保存在当前用户的数据目录。"
!insertmacro MUI_PAGE_WELCOME
!define MUI_LICENSEPAGE_TEXT_TOP "VisualMap 自有代码与第三方组件各自遵循其许可。"
!define MUI_LICENSEPAGE_BUTTON "下一步"
!insertmacro MUI_PAGE_LICENSE "DISTRIBUTION_LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\VisualMap.exe"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README_FIRST.txt"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH
!insertmacro MUI_LANGUAGE "SimpChinese"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "VisualMap 需要 64 位 Windows 10 或 Windows 11。"
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "VisualMap 需要 Windows 10 或 Windows 11。"
    Abort
  ${EndIf}
FunctionEnd

Function StopExisting
  IfFileExists "$INSTDIR\VisualMap.exe" 0 finished
  ExecWait '"$INSTDIR\VisualMap.exe" --stop' $0
  ${If} $0 != 0
  ${AndIf} $0 != 3
    MessageBox MB_ICONEXCLAMATION "VisualMap 尚有任务进行中或无法安全退出。请先完成识别与复核，再从开始菜单退出 VisualMap 后重试。"
    Abort
  ${EndIf}
  finished:
FunctionEnd

Section "VisualMap" Main
  Call StopExisting
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  SetOverwrite on
  AllowSkipFiles off
  ClearErrors
  File /r "${BUNDLE}\*"
  ${If} ${Errors}
    SetErrorLevel 2
    Abort "部分程序文件未能完整安装，请检查目录权限后重试。"
  ${EndIf}
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\VisualMap"
  CreateShortcut "$SMPROGRAMS\VisualMap\VisualMap.lnk" "$INSTDIR\VisualMap.exe"
  CreateShortcut "$SMPROGRAMS\VisualMap\退出 VisualMap.lnk" "$INSTDIR\VisualMap.exe" "--stop"
  CreateShortcut "$SMPROGRAMS\VisualMap\使用说明.lnk" "$WINDIR\System32\notepad.exe" '"$INSTDIR\README_FIRST.txt"'
  CreateShortcut "$SMPROGRAMS\VisualMap\开源与第三方说明.lnk" "$WINDIR\System32\notepad.exe" '"$INSTDIR\THIRD_PARTY_NOTICES.txt"'
  CreateShortcut "$SMPROGRAMS\VisualMap\卸载 VisualMap.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\VisualMap" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "DisplayName" "VisualMap"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "DisplayVersion" "0.1.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "Publisher" "Hyperdimensionalreference"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "DisplayIcon" "$INSTDIR\VisualMap.exe"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap" "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  IfFileExists "$INSTDIR\VisualMap.exe" 0 remove
  ExecWait '"$INSTDIR\VisualMap.exe" --stop' $0
  ${If} $0 != 0
  ${AndIf} $0 != 3
    MessageBox MB_ICONEXCLAMATION "仍有任务进行中或无法安全退出，请先退出 VisualMap 再卸载。个人数据保持。"
    Abort
  ${EndIf}
  remove:
  ; Generated from the staged bundle: no recursive deletion of the selected
  ; install directory and no deletion of LocalAppData\VisualMap user data.
  !include "${DELETE_MANIFEST}"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\VisualMap\VisualMap.lnk"
  Delete "$SMPROGRAMS\VisualMap\退出 VisualMap.lnk"
  Delete "$SMPROGRAMS\VisualMap\使用说明.lnk"
  Delete "$SMPROGRAMS\VisualMap\开源与第三方说明.lnk"
  Delete "$SMPROGRAMS\VisualMap\卸载 VisualMap.lnk"
  RMDir "$SMPROGRAMS\VisualMap"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualMap"
  DeleteRegKey HKCU "Software\VisualMap"
SectionEnd
