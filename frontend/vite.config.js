import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react' // Changed this line
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})