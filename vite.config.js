import { defineConfig } from 'vite';

// main.js awaits at the top level (renderer.init, the field)
export default defineConfig({
  build: { target: 'esnext' },
  optimizeDeps: { esbuildOptions: { target: 'esnext' } },
  server: { port: 5190, strictPort: true },
});
