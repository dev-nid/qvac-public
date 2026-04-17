'use strict'

const LlmLlamacpp = require('../index')
const FilesystemDL = require('@qvac/dl-filesystem')
const process = require('bare-process')
const { downloadModel } = require('./utils')

const MAIN_GPU_VALUES = [
  { label: 'none (default)', value: undefined },
  { label: 'integer 0', value: '0' },
  { label: 'integer 1', value: '1' },
  { label: 'integer 2', value: '2' },
  { label: '"integrated"', value: 'integrated' },
  { label: '"dedicated"', value: 'dedicated' }
]

function getTestValue () {
  const arg = process.argv.find(a => a.startsWith('--main-gpu='))
  if (!arg) return MAIN_GPU_VALUES[2]

  const raw = arg.split('=')[1]
  const match = MAIN_GPU_VALUES.find(v => v.value === raw)
  return match || { label: `custom "${raw}"`, value: raw }
}

async function testMainGpu (testCase, modelName, dirPath) {
  console.log(`\n${'='.repeat(60)}`)
  console.log(`Testing main-gpu: ${testCase.label}`)
  console.log('='.repeat(60))

  const fsDL = new FilesystemDL({ dirPath })

  const args = {
    loader: fsDL,
    opts: { stats: true },
    logger: console,
    diskPath: dirPath,
    modelName
  }

  const config = {
    device: 'gpu',
    gpu_layers: '999',
    ctx_size: '512',
    verbosity: '3'
  }

  if (testCase.value !== undefined) {
    config['main-gpu'] = testCase.value
  }

  console.log('Config:', JSON.stringify(config, null, 2))

  const model = new LlmLlamacpp(args, config)

  try {
    console.log('\nLoading model...')
    await model.load()
    console.log('Model loaded successfully')

    const prompt = [
      { role: 'user', content: 'Say hello in one sentence.' }
    ]

    console.log('Running inference...')
    const response = await model.run(prompt)
    let fullResponse = ''

    await response
      .onUpdate(data => {
        process.stdout.write(data)
        fullResponse += data
      })
      .await()

    console.log('\n')
    console.log('Inference stats:', JSON.stringify(response.stats))
    console.log(`RESULT: main-gpu=${testCase.label} => SUCCESS`)
  } catch (error) {
    const errorMessage = error?.message || error?.toString() || String(error)
    console.error(`\nERROR with main-gpu=${testCase.label}:`, errorMessage)
    console.error('Stack:', error?.stack)
    console.log(`RESULT: main-gpu=${testCase.label} => FAILED`)
  } finally {
    await model.unload()
    await fsDL.close()
  }
}

async function main () {
  console.log('main-gpu Parameter Test')
  console.log('Usage: bare examples/mainGpuTest.js [--main-gpu=<value>] [--all]')
  console.log('  --main-gpu=1          Test a specific value (0, 1, 2, integrated, dedicated)')
  console.log('  --all                 Test all values sequentially')
  console.log('')

  const [modelName, dirPath] = await downloadModel(
    'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf',
    'Llama-3.2-1B-Instruct-Q4_0.gguf'
  )

  const runAll = process.argv.includes('--all')

  if (runAll) {
    console.log('Running ALL main-gpu test cases...')
    for (const testCase of MAIN_GPU_VALUES) {
      await testMainGpu(testCase, modelName, dirPath)
    }

    console.log(`\n${'='.repeat(60)}`)
    console.log('All tests complete')
    console.log('='.repeat(60))
  } else {
    const testCase = getTestValue()
    await testMainGpu(testCase, modelName, dirPath)
  }
}

main().catch(error => {
  console.error('Fatal error:', error.message)
  console.error('Stack:', error.stack)
  process.exit(1)
})
