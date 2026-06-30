import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.relu(self.norm(self.conv(x)))
        return (out + x).transpose(1, 2)

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, dim=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, memory, tokens):
        emb = self.token_emb(tokens)
        h0 = memory.unsqueeze(0)
        out, _ = self.gru(emb, h0)
        return self.fc(out)

class AVTCNGRUModel(nn.Module):
    def __init__(self, vocab_size, hidden=512, num_classes=28):
        super().__init__()

        self.v_proj = nn.Linear(512, hidden)
        self.a_proj = nn.Linear(1024, hidden)

        self.fusion = nn.Linear(hidden * 2, hidden)

        self.tcn = nn.Sequential(
            TCNBlock(hidden, dilation=1),
            TCNBlock(hidden, dilation=2),
            TCNBlock(hidden, dilation=4)
        )

        self.action_head = nn.Linear(hidden, num_classes)

        self.caption_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.decoder = GRUDecoder(vocab_size, hidden)

    def forward(self, v_feats, a_feats, action_text_bank):
        v = self.v_proj(v_feats)
        a = self.a_proj(a_feats)

        x = self.fusion(torch.cat([v, a], dim=-1))
        h = self.tcn(x)

        pooled = h.mean(dim=1)

        logits_cls = self.action_head(pooled)

        probs = torch.softmax(logits_cls, dim=-1)
        action_emb = probs @ action_text_bank

        cond = torch.cat([pooled, action_emb], dim=-1)
        caption_emb = self.caption_head(cond)

        return logits_cls, caption_emb
