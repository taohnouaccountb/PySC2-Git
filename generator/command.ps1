python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent pysc2.agents.random_gen.RandomAgent --save_replay false   --max_agent_steps 10000000 --parallel 3

python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards

python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent pysc2.agents.random_gen.RandomAgent --save_replay false   --max_agent_steps 10000000