# Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files from web-app directory
COPY web-app/package.json ./
RUN npm install

# Copy source code from web-app directory
COPY web-app/ .

# Build the app
RUN npm run build

# Production stage
FROM node:20-alpine

WORKDIR /app

# Copy built app and dependencies from builder
COPY --from=builder /app/dist ./
COPY --from=builder /app/node_modules ./node_modules

ENV HOST=0.0.0.0
ENV PORT=3000

EXPOSE 3000

CMD ["node", "./server/entry.mjs"]