import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: true }, // accesibil și de pe alt dispozitiv din rețea (ex. tableta de la demo)
})
