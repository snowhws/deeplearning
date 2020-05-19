package main

import (
  "fmt"
  "net/http"
  "log"
)

func Response(w http.ResponseWriter, r *http.Request) {
  fmt.Fprintln(w, "hello world")
}

func main() {
  http.HandleFunc("/", Response)
  err := http.ListenAndServe(":8080", nil)
  if err != nil {
    log.Fatal("ListenAndServer: ", err.Error())
  }
}

