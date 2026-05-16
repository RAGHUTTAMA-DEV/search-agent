import { SearchAgentGraph } from "./search-agent"
import * as readline from "readline"

async function chat(query: string) {
    const threadId = crypto.randomUUID()  // one thread per conversation
    const config = { configurable: { thread_id: threadId } }

    await SearchAgentGraph.invoke(
        { query, iterations: 0 },
        config
    )

    const pausedState = await SearchAgentGraph.getState(config)
    const answer = pausedState.values.answer

    console.log(" Agent answer:\n", answer)

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
    const decision = await new Promise<string>(resolve => {
        rl.question("\nApprove this answer? (approve/retry): ", ans => {
            rl.close()
            resolve(ans.trim().toLowerCase())
        })
    })

    await SearchAgentGraph.updateState(config, {
        humanDecision: decision === "retry" ? "retry" : "approve"
    })

    const finalResult = await SearchAgentGraph.invoke(null, config)

    return finalResult.answer
}

chat("what is today's ipl match and score?")
    .then(answer => console.log("\nFinalds:", answer))
    .catch(console.error)