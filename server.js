// server.js — Robust Hybrid OpenAI ↔ NIM Proxy (Edição Sem Travamentos)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Desativando o bloqueio de CLIENT_AUTH_KEY por padrão para parar de dar 403/Failed to fetch
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY || null; 

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

const MAX_TOKENS_LIMIT = 4096; // Ajustado para maior estabilidade na Vercel
const REQUEST_TIMEOUT_MS = 60000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

// Configuração para compatibilidade com Vercel Edge se necessário
module.exports = { config: { runtime: 'edge' } };

// ─── Model Mapping ─────────────────────────────────────────────────────────
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gemini-pro': 'meta/llama-3.1-70b-instruct'
};

const FALLBACK_MODELS = ['meta/llama-3.1-70b-instruct', 'meta/llama-3.1-8b-instruct'];

// ─── Middleware ─────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Middleware de Autenticação simplificado (Não derruba o app)
app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  // Se você NÃO configurou CLIENT_AUTH_KEY nas variáveis da Vercel, ele deixa passar direto
  if (!CLIENT_AUTH_KEY) {
    return next();
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(403).json({ error: { message: 'Missing authentication token' } });
  }

  const token = authHeader.split(' ')[1];
  if (token !== CLIENT_AUTH_KEY) {
    return res.status(403).json({ error: { message: 'Invalid authentication credentials' } });
  }

  next();
});

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────
function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────
async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
    try {
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );
      return { response: res, model };
    } catch (err) {
      lastError = err;
      console.warn(`[FALLBACK] Model failed: ${model}`, err.message);
    }
  }
  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({ status: 'ok', proxy: 'running' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const primaryModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    
    if (stream) {
      upstreamStream = response.data;
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let doneSent = false;

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (let line of lines) {
          line = line.trim();
          if (!line) continue;

          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              if (!doneSent) {
                safeWrite(res, 'data: [DONE]\n\n');
                doneSent = true;
              }
              streamEndedCleanly = true;
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              const delta = data.choices?.[0]?.delta;

              if (delta && !SHOW_REASONING) {
                // Remove o nó de raciocínio se a flag estiver falsa para evitar poluição
                delete delta.reasoning_content;
              }

              safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Ignora linhas incompletas silenciosamente para não quebrar o stream
            }
          }
        }
      });

      upstreamStream.on('end', () => {
        if (!doneSent) safeWrite(res, 'data: [DONE]\n\n');
        res.end();
      });

      upstreamStream.on('error', () => res.end());
    } else {
      // Non-streaming response
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => ({
          index: i,
          message: {
            role: choice.message?.role || 'assistant',
            content: choice.message?.content || ''
          },
          finish_reason: choice.finish_reason || 'stop'
        })),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: { message: error.message } });
    }
  }
});

// 404 Handler padrão
app.use((req, res) => {
  res.status(404).json({ error: { message: `Route ${req.path} not found` } });
});

app.listen(PORT, () => {
  console.log(`[PROXY] Proxy running on port ${PORT}`);
});
