def beam_search(model, input_tensor, max_length=50, beam_size=5):
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(input_tensor, None)
        initial_beam = [{'tokens': [start_token], 'log_prob': 0.0}]

        for _ in range(max_length):
            new_beam = []
            for candidate in initial_beam:
                last_token = candidate['tokens'][-1]
                if last_token == end_token:
                    new_beam.append(candidate)
                    continue
                
                input_tensor = torch.tensor(candidate['tokens']).unsqueeze(0).to(device)
                output = model.decode(encoder_output, None, input_tensor, None)
                next_token_probs = nn.functional.log_softmax(output, dim=-1).squeeze()
                topk_probs, topk_indices = next_token_probs.topk(beam_size)

                for i in range(beam_size):
                    new_candidate = candidate.copy()
                    new_candidate['tokens'].append(topk_indices[i].item())
                    new_candidate['log_prob'] += topk_probs[i].item()
                    new_beam.append(new_candidate)

            new_beam.sort(key=lambda x: x['log_prob'], reverse=True)
            initial_beam = new_beam[:beam_size]

            if all(candidate['tokens'][-1] == end_token for candidate in initial_beam):
                break

        return initial_beam[0]['tokens']