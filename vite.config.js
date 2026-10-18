import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// main.js awaits at the top level (renderer.init, the field)
// HTTPS with a self-signed certificate, served on the local network too
// (host: true): WebGPU needs a secure context, which localhost is but a LAN
// address over plain HTTP is not.
// Relative base so the build runs from any path, like GitHub Pages'
// /a-crown-of-ice-and-fire/; asset URLs in src go through BASE_URL.
export default defineConfig({
  base: './',
  plugins: [basicSsl()],
  build: { target: 'esnext' },
  optimizeDeps: { esbuildOptions: { target: 'esnext' } },
  server: { port: 5190, strictPort: true, host: true },
});
