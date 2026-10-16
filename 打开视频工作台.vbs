Option Explicit
Dim files, shell, root, command
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
command = Chr(34) & root & "\.venv\Scripts\pythonw.exe" & Chr(34) & " -X utf8 " & Chr(34) & root & "\scripts\open_workspace.pyw" & Chr(34)
shell.Run command, 0, False
