"use client"

import { createContext, useContext, useState, ReactNode } from "react"

interface ModalContextType {
  isUploadModalOpen: boolean
  openUploadModal: () => void
  closeUploadModal: () => void
}

const ModalContext = createContext<ModalContextType | undefined>(undefined)

export function ModalProvider({ children }: { children: ReactNode }) {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)

  const openUploadModal = () => setIsUploadModalOpen(true)
  const closeUploadModal = () => setIsUploadModalOpen(false)

  return (
    <ModalContext.Provider 
      value={{ 
        isUploadModalOpen, 
        openUploadModal, 
        closeUploadModal 
      }}
    >
      {children}
    </ModalContext.Provider>
  )
}

export function useModal() {
  const context = useContext(ModalContext)
  if (context === undefined) {
    throw new Error("useModal must be used within a ModalProvider")
  }
  return context
}