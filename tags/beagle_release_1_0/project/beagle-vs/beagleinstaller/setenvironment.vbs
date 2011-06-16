Set WshShell = CreateObject("WScript.Shell") 
Set WshEnv = WshShell.Environment("SYSTEM") 
WshEnv("Path") = WshEnv("Path") & ";%LIBHMSBEAGLE-1.0%"