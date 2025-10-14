"use client"

import { createContext, useContext, useEffect, useState, ReactNode } from "react"
import { logout } from "@/lib/api"

interface User {
  username: string
  email: string
  full_name: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  login: (token: string, userData: User) => void
  logout: () => void
  checkAuth: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  const checkAuth = () => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("auth_token")
      const refreshToken = localStorage.getItem("refresh_token")
      const userData = localStorage.getItem("user_data")
      
      if (token && refreshToken && userData) {
        try {
          const parsedUser = JSON.parse(userData)
          setUser(parsedUser)
          setIsAuthenticated(true)
        } catch (error) {
          // Invalid user data, clear storage
          localStorage.removeItem("auth_token")
          localStorage.removeItem("refresh_token")
          localStorage.removeItem("user_data")
          setUser(null)
          setIsAuthenticated(false)
        }
      } else {
        setUser(null)
        setIsAuthenticated(false)
      }
    }
  }

  const loginUser = (token: string, userData: User) => {
    if (typeof window !== "undefined") {
      localStorage.setItem("auth_token", token)
      localStorage.setItem("user_data", JSON.stringify(userData))
    }
    setUser(userData)
    setIsAuthenticated(true)
  }

  const logoutUser = async () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("auth_token")
      localStorage.removeItem("refresh_token")
      localStorage.removeItem("user_data")
    }
    setUser(null)
    setIsAuthenticated(false)
    await logout()
  }

  useEffect(() => {
    checkAuth()
  }, [])

  return (
    <AuthContext.Provider 
      value={{ 
        user, 
        isAuthenticated, 
        login: loginUser, 
        logout: logoutUser, 
        checkAuth 
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}