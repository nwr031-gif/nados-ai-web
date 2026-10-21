# Nados AI — حاوية تشغيل
FROM node:22-alpine

WORKDIR /app

# لا تبعيات npm — ننسخ المصدر فقط
COPY package.json ./
COPY server.mjs ./
COPY nados-engine.mjs ./
COPY web-search.mjs ./
COPY lib ./lib
COPY public ./public
COPY training ./training
COPY benchmark ./benchmark
COPY tests ./tests
COPY diagnose.mjs ./

# ملفات env تُركَّب وقت التشغيل (لا تُنسخ إلى الصورة)
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=4000

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:4000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

USER node

CMD ["node", "server.mjs"]
