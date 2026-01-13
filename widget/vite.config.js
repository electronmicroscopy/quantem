import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [anywidget(), react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: "src/quantem/widget/static",
    lib: {
      entry: {
        show4dstem: "js/show4dstem.tsx",
      },
      formats: ["es"],
    },
    rollupOptions: {
      output: {
        // Each entry gets its own file
        entryFileNames: "[name].js",
        // CSS is handled separately by anywidget
        assetFileNames: "[name][extname]",
      },
    },
  },
});
