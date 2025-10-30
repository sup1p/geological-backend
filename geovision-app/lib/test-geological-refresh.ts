// Test function to simulate token expiration and refresh for geological section creation
// Add this to browser console to test the refresh logic

async function testGeologicalSectionRefresh() {
  console.log("=== Testing Geological Section Token Refresh ===")
  
  // Check if user is logged in
  const currentToken = localStorage.getItem("auth_token")
  const refreshToken = localStorage.getItem("refresh_token")
  
  if (!currentToken || !refreshToken) {
    console.error("❌ Please log in first to test this functionality")
    return
  }
  
  console.log("✅ User is logged in")
  console.log("Current token (first 20 chars):", currentToken.substring(0, 20) + "...")
  console.log("Refresh token exists:", !!refreshToken)
  
  // Step 1: Test with valid token first
  console.log("\n--- Step 1: Testing with current valid token ---")
  try {
    const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000"
    const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
      headers: {
        Authorization: `Bearer ${currentToken}`,
      },
    })
    
    if (response.ok) {
      console.log("✅ Current token is valid")
    } else {
      console.log("⚠️ Current token seems invalid:", response.status)
    }
  } catch (error) {
    console.error("❌ Error testing current token:", error)
  }
  
  // Step 2: Simulate expired token by corrupting it
  console.log("\n--- Step 2: Simulating expired token ---")
  const originalToken = currentToken
  const expiredToken = currentToken + "EXPIRED"
  localStorage.setItem("auth_token", expiredToken)
  
  console.log("🔧 Temporarily set expired token")
  
  // Step 3: Try to use authenticatedFetch (should trigger refresh)
  console.log("\n--- Step 3: Testing authenticatedFetch with expired token ---")
  try {
    const { authenticatedFetch } = window
    
    if (!authenticatedFetch) {
      console.error("❌ authenticatedFetch not available in window object")
      console.log("💡 You can manually import and test the createGeologicalSection function")
      return
    }
    
    console.log("🚀 Calling authenticatedFetch with expired token...")
    const testResponse = await authenticatedFetch(`${API_BASE_URL}/api/auth/me`)
    
    if (testResponse.ok) {
      console.log("✅ Request succeeded after token refresh!")
      console.log("New token (first 20 chars):", localStorage.getItem("auth_token")?.substring(0, 20) + "...")
    } else {
      console.log("❌ Request failed even after refresh attempt:", testResponse.status)
    }
    
  } catch (error) {
    console.error("❌ authenticatedFetch failed:", error)
    
    // Restore original token
    localStorage.setItem("auth_token", originalToken)
    console.log("🔧 Restored original token")
  }
  
  console.log("\n=== Test Complete ===")
}

// Test createGeologicalSection with mock data
async function testCreateGeologicalSectionRefresh() {
  console.log("=== Testing createGeologicalSection Token Refresh ===")
  
  // Create mock files
  const mockFile = new File(['mock content'], 'test.png', { type: 'image/png' })
  
  console.log("📁 Created mock files for testing")
  
  // Get current token and corrupt it
  const originalToken = localStorage.getItem("auth_token")
  if (!originalToken) {
    console.error("❌ Please log in first")
    return
  }
  
  // Temporarily corrupt the token to simulate expiration
  const expiredToken = originalToken + "EXPIRED"
  localStorage.setItem("auth_token", expiredToken)
  console.log("🔧 Set expired token to test refresh logic")
  
  try {
    // This should fail with the expired token, then automatically refresh and retry
    console.log("🚀 Attempting to create geological section with expired token...")
    
    const result = await createGeologicalSection(
      mockFile, // mapImage
      mockFile, // legendImage  
      mockFile, // columnImage
      0, // startX
      0, // startY
      100, // endX
      100  // endY
    )
    
    console.log("✅ createGeologicalSection succeeded after token refresh!")
    
  } catch (error) {
    console.log("📊 createGeologicalSection result:", error.message)
    
    if (error.message.includes("UNAUTHORIZED")) {
      console.log("⚠️ Got UNAUTHORIZED error - refresh likely failed")
    } else if (error.message.includes("Failed to create section")) {
      console.log("✅ Got creation error (not auth error) - refresh likely worked but creation failed due to mock data")
    } else {
      console.log("❓ Unexpected error:", error.message)
    }
    
  } finally {
    // Restore original token
    localStorage.setItem("auth_token", originalToken)
    console.log("🔧 Restored original token")
  }
  
  console.log("\n=== Test Complete ===")
}

// Make functions available globally
if (typeof window !== 'undefined') {
  window.testGeologicalSectionRefresh = testGeologicalSectionRefresh
  window.testCreateGeologicalSectionRefresh = testCreateGeologicalSectionRefresh
  
  console.log("🧪 Test functions loaded!")
  console.log("Run: testGeologicalSectionRefresh()")
  console.log("Run: testCreateGeologicalSectionRefresh()")
}

export { testGeologicalSectionRefresh, testCreateGeologicalSectionRefresh }