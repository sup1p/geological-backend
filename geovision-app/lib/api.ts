// API client for GeoVision backend

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000"

export interface UserSection {
  id: number
  user_id: number
  section_url: string
  created_at: string
}

export interface PaginatedSections {
  items: UserSection[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface CreateSectionResponse {
  id: number
  user_id: number
  section_url: string
  created_at: string
}

export interface User {
  id: number
  username: string
  email: string
  full_name: string
}

export interface UserCreate {
  username: string
  email: string
  full_name: string
  password: string
}

export interface Token {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface LoginCredentials {
  username: string
  password: string
}

// Get authentication token from localStorage or cookies
export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null
  const token = localStorage.getItem("auth_token")
  console.log("Getting auth token:", token ? "Token exists" : "No token found")
  return token
}

// Get refresh token from localStorage
export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem("refresh_token")
}

// Store tokens in localStorage
export function storeTokens(accessToken: string, refreshToken: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem("auth_token", accessToken)
    localStorage.setItem("refresh_token", refreshToken)
  }
}

// Refresh access token using refresh token
export async function refreshAccessToken(): Promise<Token> {
  const refreshToken = getRefreshToken()
  if (!refreshToken) {
    throw new Error("No refresh token available")
  }

  const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })

  if (!response.ok) {
    // If refresh fails, remove tokens and throw error
    if (typeof window !== "undefined") {
      localStorage.removeItem("auth_token")
      localStorage.removeItem("refresh_token")
    }
    throw new Error("Failed to refresh token")
  }

  const tokenData = await response.json()
  storeTokens(tokenData.access_token, tokenData.refresh_token)
  return tokenData
}

// Enhanced fetch function that handles token refresh automatically
export async function authenticatedFetch(
  url: string, 
  options: RequestInit = {}
): Promise<Response> {
  const token = getAuthToken()
  
  if (!token) {
    throw new Error("Authentication required")
  }

  // Add Authorization header
  const headers = {
    ...options.headers,
    Authorization: `Bearer ${token}`,
  }

  const response = await fetch(url, { ...options, headers })

  // If token expired, try to refresh and retry
  if (response.status === 401) {
    try {
      await refreshAccessToken()
      const newToken = getAuthToken()
      
      if (newToken) {
        const retryHeaders = {
          ...options.headers,
          Authorization: `Bearer ${newToken}`,
        }
        return await fetch(url, { ...options, headers: retryHeaders })
      }
    } catch (refreshError) {
      // Refresh failed, redirect to login or throw error
      throw new Error("Authentication failed")
    }
  }

  return response
}

// Create enhanced geological section
export async function createGeologicalSection(
  mapImage: File,
  legendImage: File,
  columnImage: File,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
): Promise<Blob> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("Authentication required")
  }

  const formData = new FormData()
  formData.append("map_image", mapImage)
  formData.append("legend_image", legendImage)
  formData.append("column_image", columnImage)
  formData.append("start_x", startX.toString())
  formData.append("start_y", startY.toString())
  formData.append("end_x", endX.toString())
  formData.append("end_y", endY.toString())

  const response = await fetch(`${API_BASE_URL}/api/v1/geological-section/create-enhanced-section`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`
    try {
      const errorText = await response.text()
      if (errorText) {
        errorMessage = errorText
      }
    } catch (e) {
      console.error('Error reading response:', e)
    }
    
    if (response.status === 401) {
      throw new Error("Authentication failed. Please log in again.")
    }
    
    throw new Error(`Failed to create section: ${errorMessage}`)
  }

  return response.blob()
}

// Get user sections with pagination
export async function getUserSections(page = 1, pageSize = 10): Promise<PaginatedSections> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("Authentication required")
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/sections/?page=${page}&page_size=${pageSize}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (!response.ok) {
    throw new Error("Failed to fetch sections")
  }

  return response.json()
}

// Get specific user section
export async function getUserSection(sectionId: number): Promise<UserSection> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("Authentication required")
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/sections/${sectionId}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (!response.ok) {
    throw new Error("Failed to fetch section")
  }

  return response.json()
}

// Delete user section
export async function deleteUserSection(sectionId: number): Promise<void> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("Authentication required")
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/sections/${sectionId}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (!response.ok) {
    throw new Error("Failed to delete section")
  }
}

// Authentication functions
export async function register(userData: UserCreate): Promise<User> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Registration failed")
    }

    return response.json()
  } catch (error) {
    if (error instanceof Error) {
      throw error
    }
    throw new Error("Server is not responding. Please try again later.")
  }
}

export async function login(credentials: LoginCredentials): Promise<{ token: Token; user: User }> {
  try {
    const formData = new FormData()
    formData.append("username", credentials.username)
    formData.append("password", credentials.password)

    const response = await fetch(`${API_BASE_URL}/api/auth/token`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error("Incorrect username or password")
      }
      const error = await response.json()
      throw new Error(error.detail || "Login failed")
    }

    const tokenData = await response.json()
    console.log("Login successful, tokens received:", 
      tokenData.access_token ? "Access token: Yes" : "Access token: No",
      tokenData.refresh_token ? ", Refresh token: Yes" : ", Refresh token: No")
    
    // Store both tokens
    storeTokens(tokenData.access_token, tokenData.refresh_token)
    
    // Get user data using the token
    const userResponse = await fetch(`${API_BASE_URL}/api/auth/me`, {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
      },
    })

    let userData: User
    if (userResponse.ok) {
      userData = await userResponse.json()
    } else {
      // Fallback: create user data from email
      userData = {
        id: 0,
        username: credentials.username,
        email: credentials.username,
        full_name: credentials.username
      }
    }

    return { token: tokenData, user: userData }
  } catch (error) {
    if (error instanceof Error) {
      throw error
    }
    throw new Error("Server is not responding. Please try again later.")
  }
}

export async function logout(): Promise<void> {
  if (typeof window !== "undefined") {
    localStorage.removeItem("auth_token")
    localStorage.removeItem("refresh_token")
  }
}

// WebSocket connection for AI chatbot
export function createWebSocketConnection(token: string): WebSocket {
  if (!token) {
    throw new Error("Token is required for WebSocket connection")
  }
  
  const wsUrl = API_BASE_URL.replace("http", "ws")
  const websocketUrl = `${wsUrl}/api/v1/ws/strato?token=${encodeURIComponent(token)}`
  
  console.log("Creating WebSocket connection to:", websocketUrl.replace(token, '***'))
  
  return new WebSocket(websocketUrl)
}
