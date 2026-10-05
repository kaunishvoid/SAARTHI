import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiTarget = process.env.SAARTHI_API_ORIGIN || 'http://127.0.0.1:5000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': { target: apiTarget, changeOrigin: true },
      '/api': { target: apiTarget, changeOrigin: true },
    },
  },
})
