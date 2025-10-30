// Quick test script to verify refresh token logic
// Run this in browser console or create as a separate test file

// Test the refresh token API endpoint
async function testRefreshToken() {
  console.log("Testing refresh token logic...")
  
  // Get current tokens from localStorage
  const currentRefreshToken = localStorage.getItem("refresh_token")
  const currentAccessToken = localStorage.getItem("auth_token")
  
  console.log("Current tokens:")
  console.log("Access token exists:", !!currentAccessToken)
  console.log("Refresh token exists:", !!currentRefreshToken)
  
  if (!currentRefreshToken) {
    console.error("No refresh token found. Please login first.")
    return
  }
  
  try {
    // Test the refresh endpoint
    const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000"
    
    console.log("Making refresh request...")
    const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh_token: currentRefreshToken }),
    })
    
    if (!response.ok) {
      console.error("Refresh failed:", response.status, response.statusText)
      const errorText = await response.text()
      console.error("Error body:", errorText)
      return
    }
    
    const tokenData = await response.json()
    console.log("Refresh successful!")
    console.log("New tokens received:", {
      access_token_length: tokenData.access_token?.length,
      refresh_token_length: tokenData.refresh_token?.length,
      token_type: tokenData.token_type
    })
    
    // Store the new tokens
    localStorage.setItem("auth_token", tokenData.access_token)
    localStorage.setItem("refresh_token", tokenData.refresh_token)
    
    console.log("New tokens stored successfully")
    
    // Test with the new access token
    console.log("Testing new access token with /api/auth/me...")
    const meResponse = await fetch(`${API_BASE_URL}/api/auth/me`, {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
      },
    })
    
    if (meResponse.ok) {
      const userData = await meResponse.json()
      console.log("New access token works! User data:", userData)
    } else {
      console.error("New access token failed:", meResponse.status, meResponse.statusText)
    }
    
  } catch (error) {
    console.error("Test failed with error:", error)
  }
}

// Test WebSocket authentication error handling
async function testWebSocketAuthError() {
  console.log("Testing WebSocket auth error handling...")
  
  // This would need to be run when connected to WebSocket
  console.log("To test WebSocket auth errors:")
  console.log("1. Open the chatbot")
  console.log("2. Wait for connection")
  console.log("3. Manually expire the access token or send invalid token")
  console.log("4. Send a message")
  console.log("5. Server should respond with {\"error\": \"Authentication failed\"}")
  console.log("6. Frontend should automatically try to refresh the token")
}

// Export test functions
if (typeof window !== 'undefined') {
  // Browser environment
  window.testRefreshToken = testRefreshToken
  window.testWebSocketAuthError = testWebSocketAuthError
  
  console.log("Refresh token test functions loaded!")
  console.log("Run testRefreshToken() to test refresh logic")
  console.log("Run testWebSocketAuthError() for WebSocket test info")
}

export { testRefreshToken, testWebSocketAuthError }