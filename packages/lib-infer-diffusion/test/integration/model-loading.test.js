'use strict'

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const proc = require('bare-process')

const ImgStableDiffusion = require('../../index.js')
const { ensureModel } = require('./utils')

const platform = os.platform()
const arch = os.arch()
const isDarwinX64 = platform === 'darwin' && arch === 'x64'
const isLinuxArm64 = platform === 'linux' && arch === 'arm64'
const noGpu = proc.env && proc.env.NO_GPU === 'true'
const useCpu = isDarwinX64 || isLinuxArm64 || noGpu

const DEFAULT_MODEL = {
  name: 'stable-diffusion-v2-1-Q8_0.gguf',
  url: 'https://huggingface.co/gpustack/stable-diffusion-v2-1-GGUF/resolve/main/stable-diffusion-v2-1-Q8_0.gguf'
}

test('model loading - load and unload', { timeout: 600_000 }, async t => {
  const [downloadedModelName, modelDir] = await ensureModel({
    modelName: DEFAULT_MODEL.name,
    downloadUrl: DEFAULT_MODEL.url
  })

  const config = {
    threads: '4',
    device: useCpu ? 'cpu' : 'gpu',
    prediction: 'v'
  }

  const addon = new ImgStableDiffusion({
    modelName: downloadedModelName,
    diskPath: modelDir,
    logger: console
  }, config)

  await addon.load()
  t.pass('model loaded successfully')

  await addon.unload()
  t.pass('model unloaded successfully')

  await addon.unload().catch(() => {})
  t.pass('second unload is idempotent')
})

const isAndroid = platform === 'android'

test('opencl cache - second load is faster than first', { timeout: 600_000, skip: !isAndroid }, async t => {
  const [downloadedModelName, modelDir] = await ensureModel({
    modelName: DEFAULT_MODEL.name,
    downloadUrl: DEFAULT_MODEL.url
  })

  const cacheDir = path.join(modelDir, 'opencl-cache')
  if (fs.existsSync(cacheDir)) {
    fs.rmSync(cacheDir, { recursive: true })
  }

  const config = {
    threads: '4',
    device: 'gpu',
    prediction: 'v',
    openclCacheDir: modelDir
  }

  const args = {
    modelName: downloadedModelName,
    diskPath: modelDir,
    logger: console
  }

  const first = new ImgStableDiffusion(args, config)
  const t0 = Date.now()
  await first.load()
  const coldLoadMs = Date.now() - t0
  console.log(`Cold load (no cache): ${coldLoadMs} ms`)
  await first.unload()

  t.ok(fs.existsSync(cacheDir), 'opencl-cache directory was created')
  const cachedFiles = fs.readdirSync(cacheDir).filter(f => f.endsWith('.oclbin'))
  t.ok(cachedFiles.length > 0, `Cache contains ${cachedFiles.length} .oclbin files`)

  const second = new ImgStableDiffusion(args, config)
  const t1 = Date.now()
  await second.load()
  const warmLoadMs = Date.now() - t1
  console.log(`Warm load (cached): ${warmLoadMs} ms`)
  await second.unload()

  console.log(`Speedup: ${(coldLoadMs / warmLoadMs).toFixed(1)}x`)
  t.ok(warmLoadMs < coldLoadMs, `Warm load (${warmLoadMs} ms) faster than cold load (${coldLoadMs} ms)`)
})

// Keep event loop alive briefly to let pending async operations complete
// This prevents C++ destructors from running while async cleanup is still happening
// which can cause segfaults (exit code 139)
setImmediate(() => {
  setTimeout(() => {}, 500)
})
