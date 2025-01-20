import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGuidedSemanticRefinement(nn.Module):
    def __init__(self, image_encoder, text_encoder, mask_generator, lambda_refine=0.05):
        super().__init__()
        self.E_I = image_encoder
        self.E_T = text_encoder
        self.S = mask_generator
        self.lambda_refine = lambda_refine

    def forward(self, X, p_k, t_k):
        """
        - X = (B, C, H, W)
        - p_k = background prompt (B, prompt_dim)
        - t_k = text prompt embeddings (B, prompt_dim)
        """
        # Produce refined masks M'
        M_prime = self.S(X)

        #  derive the foreground image Xfk = Mk · X by applying the kth predicted mask Mk to the original image X
        X_f_k = M_prime * X
        # vfk = EI (Xfk )
        v_f_k = self.E_I(X_f_k)
        u_b_k = self.E_T(p_k)

        # refinement loss
        L_refine = -torch.log(1 - F.cosine_similarity(v_f_k, u_b_k, dim=-1)).mean()

        L_match =

        L_total = L_refine + self.lambda_refine * L_match
        return L_total, M_prime, X_f_k

