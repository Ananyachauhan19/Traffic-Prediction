import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// ESM config (.mjs) to avoid using Vite's CJS Node API which is deprecated.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
