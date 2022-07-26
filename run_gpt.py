from gpt_generate import gpt_generate
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text", type=str, default="Today is a nice day")
    argparser.add_argument("--txt_path", type=str)
    argparser.add_argument("--stop_token", type=str, default="\n")
    argparser.add_argument("--stop_completion_on_token", action="store_true")
    argparser.add_argument("--num_return_sequences", type=int, default=1)
    argparser.add_argument("--gpu", action="store_true")
    argparser.add_argument("--with_log_probs", action="store_true")
    argparser.add_argument("--max_length", type=int, default=50)
    argparser.add_argument("--no_outputs", action="store_true")
    argparser.add_argument("--time_test", action="store_true")
    argparser.add_argument("--save_completions", action="store_true")

    args = argparser.parse_args()

    gpt_generate(
        text=args.text,
        txt_path=args.txt_path,
        stop_token=args.stop_token,
        stop_completion_on_token=args.stop_completion_on_token,
        num_return_sequences=args.num_return_sequences,
        gpu=args.gpu,
        with_log_probs=args.with_log_probs,
        max_length=args.max_length,
        no_outputs=args.no_outputs,
        time_test=args.time_test,
        save_completions=args.save_completions,
    )
