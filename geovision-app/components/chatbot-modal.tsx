"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { X, Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { createWebSocketConnection } from "@/lib/api"
import { UnauthorizedError } from "@/components/ui/unauthorized-error"

// Loading dots animation component
function LoadingDots() {
  return (
    <div className="flex items-center space-x-1">
      <span>Strato is thinking</span>
      <div className="flex space-x-1">
        <div className="w-1 h-1 bg-current rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
        <div className="w-1 h-1 bg-current rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
        <div className="w-1 h-1 bg-current rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  )
}

// Custom hook for typewriter effect
function useTypewriter(text: string, speed: number = 30) {
  const [displayText, setDisplayText] = useState("")
  const [isTypingComplete, setIsTypingComplete] = useState(false)

  useEffect(() => {
    if (!text) return

    setDisplayText("")
    setIsTypingComplete(false)
    
    let i = 0
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayText((prev) => prev + text.charAt(i))
        i++
      } else {
        setIsTypingComplete(true)
        clearInterval(timer)
      }
    }, speed)

    return () => clearInterval(timer)
  }, [text, speed])

  return { displayText, isTypingComplete }
}

// Message bubble component with typewriter effect
function MessageBubble({ message }: { message: Message }) {
  const { displayText } = useTypewriter(message.isTyping ? message.text : "", 30)
  
  const messageText = message.isLoading ? "" : message.isTyping ? displayText : message.text

  return (
    <div className={cn("flex", message.sender === "user" ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-lg px-3 py-2",
          message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
      >
        {message.isLoading ? (
          <LoadingDots />
        ) : (
          <p className="text-sm">
            {messageText}
            {message.isTyping && <span className="animate-pulse">|</span>}
          </p>
        )}
        <p className="mt-1 text-xs opacity-70">{message.timestamp.toLocaleTimeString()}</p>
      </div>
    </div>
  )
}

interface ChatbotModalProps {
  isOpen: boolean
  onClose: () => void
}

interface Message {
  id: string
  text: string
  sender: "user" | "bot"
  timestamp: Date
  isLoading?: boolean
  isTyping?: boolean
}

export function ChatbotModal({ isOpen, onClose }: ChatbotModalProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! I'm Strato, your AI geological assistant. How can I help you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false)
  const [showUnauthorizedError, setShowUnauthorizedError] = useState(false)
  const [tokenRefreshTrigger, setTokenRefreshTrigger] = useState(0) // Trigger for reconnection after token refresh
  const wsRef = useRef<WebSocket | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen && !isConnecting && !isConnected) {
      const token = localStorage.getItem("auth_token")
      
      if (!token) {
        console.error("No authentication token found")
        return
      }

      setIsConnecting(true)

      try {
        const ws = createWebSocketConnection(
          token, 
          () => {
            // Called when unauthorized message is received and refresh failed
            setShowUnauthorizedError(true)
            setIsConnected(false)
            setIsConnecting(false)
          },
          (newToken) => {
            // Called when token was refreshed successfully
            console.log("[Chatbot] Token refreshed, reconnecting...")
            setIsConnected(false)
            setIsConnecting(false)
            // Trigger reconnection by incrementing trigger
            setTokenRefreshTrigger(prev => prev + 1)
          }
        )

        ws.onopen = () => {
          setIsConnected(true)
          setIsConnecting(false)
          console.log("[v0] WebSocket connected")
        }

        ws.onmessage = (event) => {
          // Check if this is an authentication error that should be handled by our auth handler
          try {
            const parsed = JSON.parse(event.data)
            if (parsed.error === "Authentication failed") {
              // This will be handled by the auth handler in createWebSocketConnection
              return
            }
          } catch (parseError) {
            // Not JSON, continue with normal message processing
          }
          
          // Remove loading message and add typing message
          setMessages((prev) => {
            const filteredMessages = prev.filter(msg => !msg.isLoading)
            const botMessage: Message = {
              id: Date.now().toString(),
              text: event.data,
              sender: "bot",
              timestamp: new Date(),
              isTyping: true,
            }
            return [...filteredMessages, botMessage]
          })
          setIsWaitingForResponse(false)
        }

        ws.onerror = (error) => {
          console.error("[v0] WebSocket error:", error)
          setIsConnected(false)
          setIsConnecting(false)
        }

        ws.onclose = (event) => {
          setIsConnected(false)
          setIsConnecting(false)
          console.log("[v0] WebSocket disconnected:", event.code, event.reason)
          
          // Only log the disconnection, don't automatically reconnect
          if (event.code !== 1000 && event.code !== 1001) {
            console.warn("Unexpected WebSocket disconnection")
          }
        }

        wsRef.current = ws
      } catch (error) {
        console.error("[v0] Failed to create WebSocket:", error)
        setIsConnected(false)
        setIsConnecting(false)
      }
    }

    // Close connection when modal is closed
    if (!isOpen && wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
      setIsConnected(false)
      setIsConnecting(false)
    }

    return () => {
      if (!isOpen && wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
        setIsConnected(false)
        setIsConnecting(false)
      }
    }
  }, [isOpen, isConnecting, isConnected, tokenRefreshTrigger])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || !wsRef.current || isWaitingForResponse) return

    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: "user",
      timestamp: new Date(),
    }

    // Add loading message immediately
    const loadingMessage: Message = {
      id: `loading-${Date.now()}`,
      text: "",
      sender: "bot",
      timestamp: new Date(),
      isLoading: true,
    }

    setMessages((prev) => [...prev, userMessage, loadingMessage])
    setIsWaitingForResponse(true)
    wsRef.current.send(input)
    setInput("")
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleReconnect = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setIsConnecting(false)
    
    // Trigger reconnection
    setTimeout(() => {
      const token = localStorage.getItem("auth_token")
      if (!token) {
        console.error("No authentication token found")
        return
      }

      setIsConnecting(true)

      try {
        const ws = createWebSocketConnection(
          token, 
          () => {
            // Called when unauthorized message is received during reconnect and refresh failed
            setShowUnauthorizedError(true)
            setIsConnected(false)
            setIsConnecting(false)
          },
          (newToken) => {
            // Called when token was refreshed successfully during reconnect
            console.log("[Chatbot] Token refreshed during reconnect, reconnecting...")
            setIsConnected(false)
            setIsConnecting(false)
            // Trigger another reconnection with new token
            setTokenRefreshTrigger(prev => prev + 1)
          }
        )

        ws.onopen = () => {
          setIsConnected(true)
          setIsConnecting(false)
          console.log("[v0] WebSocket reconnected")
        }

        ws.onmessage = (event) => {
          // Check if this is an authentication error that should be handled by our auth handler
          try {
            const parsed = JSON.parse(event.data)
            if (parsed.error === "Authentication failed") {
              // This will be handled by the auth handler in createWebSocketConnection
              return
            }
          } catch (parseError) {
            // Not JSON, continue with normal message processing
          }
          
          // Remove loading message and add typing message
          setMessages((prev) => {
            const filteredMessages = prev.filter(msg => !msg.isLoading)
            const botMessage: Message = {
              id: Date.now().toString(),
              text: event.data,
              sender: "bot",
              timestamp: new Date(),
              isTyping: true,
            }
            return [...filteredMessages, botMessage]
          })
          setIsWaitingForResponse(false)
        }

        ws.onerror = (error) => {
          console.error("[v0] WebSocket error:", error)
          setIsConnected(false)
          setIsConnecting(false)
        }

        ws.onclose = (event) => {
          setIsConnected(false)
          setIsConnecting(false)
          console.log("[v0] WebSocket disconnected:", event.code, event.reason)
        }

        wsRef.current = ws
      } catch (error) {
        console.error("[v0] Failed to reconnect WebSocket:", error)
        setIsConnected(false)
        setIsConnecting(false)
      }
    }, 100)
  }

  if (!isOpen) return null

  // Show unauthorized error if needed
  if (showUnauthorizedError) {
    return (
      <UnauthorizedError 
        message="Your session has expired. Please log in again to continue chatting with Strato."
        onClose={() => {
          setShowUnauthorizedError(false)
          onClose()
        }}
      />
    )
  }

  return (
    <>
      {/* Invisible overlay to detect clicks outside */}
      <div className="fixed inset-0 z-40" onClick={onClose} />
      
      <div className="fixed bottom-20 right-6 z-50 flex h-[500px] w-80 flex-col rounded-lg border border-border bg-card shadow-2xl animate-in slide-in-from-right duration-300">
      <div className="flex items-center justify-between border-b border-border p-3">
        <div>
          <h2 className="text-base font-semibold">Ask Strato</h2>
          <p className="text-xs text-muted-foreground">
            {isConnected ? "Connected" : isConnecting ? "Connecting..." : "Disconnected"}
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </Button>
      </div>

      <ScrollArea className="flex-1 p-3" ref={scrollRef}>
        <div className="space-y-3">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>
      </ScrollArea>

      <div className="border-t border-border p-3">
        {!isConnected && !isConnecting && (
          <div className="mb-2 text-center">
            <Button onClick={handleReconnect} variant="outline" size="sm">
              Reconnect
            </Button>
          </div>
        )}
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about geology..."
            className="flex-1 text-sm"
            disabled={!isConnected || isWaitingForResponse}
          />
          <Button onClick={handleSend} disabled={!isConnected || !input.trim() || isWaitingForResponse} size="sm">
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </div>
      </div>
    </>
  )
}
