"use client"

import type React from "react"

import { useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { Header } from "./header"
import { Sidebar } from "./sidebar"
import { UploadModal } from "./upload-modal"
import { ChatbotModal } from "./chatbot-modal"
import { AskStratoButton } from "./ask-strato-button"
import { useModal } from "@/contexts/modal-context"
import { useAuth } from "@/contexts/auth-context"

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isChatbotOpen, setIsChatbotOpen] = useState(false)
  const { isUploadModalOpen, openUploadModal, closeUploadModal } = useModal()
  const { isAuthenticated } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const handleAskStratoClick = () => {
    if (!isAuthenticated) {
      router.push("/login")
    } else {
      setIsChatbotOpen(true)
    }
  }

  // Hide Ask Strato button on login and signup pages
  const hideAskStratoButton = pathname === "/login" || pathname === "/signup"

  return (
    <>
      <Header onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        onOpenUploadModal={openUploadModal}
      />
      <UploadModal 
        isOpen={isUploadModalOpen} 
        onClose={closeUploadModal} 
      />
      <ChatbotModal 
        isOpen={isChatbotOpen} 
        onClose={() => setIsChatbotOpen(false)} 
      />
      {!hideAskStratoButton && !isChatbotOpen && (
        <AskStratoButton 
          onClick={handleAskStratoClick} 
          disabled={sidebarOpen}
          dimmed={sidebarOpen}
        />
      )}
      {children}
    </>
  )
}
