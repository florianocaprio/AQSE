import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 3000,
      // Permit only the Compose service alias used by backend acceptance checks.
      allowedHosts: ["frontend"],
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
