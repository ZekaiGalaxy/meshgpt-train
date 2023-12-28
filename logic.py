# Input:
# N vertices, F faces, B as batch size
class MeshAutoEncoder:
    def encode(
        vertices, # [B,V,3]
        faces, # [B,F,3]
        face_edges,
        face_mask,
        face_edges_mask,
        return_face_coordinates = False
    ):
        return (
            face_embed, # [B,F,d]
            discrete_face_coords # [B,F,9]
        )
    
    def quantize(
        faces, # [B,F,3]
        face_mask,
        face_embed, # [B,F,d]
    ):
        return (
            face_embed_output, # [B,F,3*d]
            codes_output, # [B,F*3,2]
            commit_loss
        )
        
    def decode(
        quantized, # [B,F,3*d] 
        face_mask
    ):
        return decoded # [B,F,3*d]
    
    def decode_from_codes_to_faces(
        codes, # [B,F*3,2]
        face_mask,
        return_discrete_codes = False
    ):
        return (
            continuous_coors, # [B,F,3,3]
            pred_face_coords, # [B,F,3,3]
            face_mask
        )
    
    def tokenize(
        vertices, # [B,V,3]
        faces, # [B,F,3]
    ):
        return codes # []
    
    def forward(
        vertices, # [B,V,3]
        faces, # [B,F,3]
        return_codes = False,
        return_recon_faces = False
    ):
        encoded, face_coordinates = encode(
            vertices,
            faces
        )

        quantized, codes, commit_loss = quantize(
            encoded,
            faces
        )

        decoded = decode(
            qunatized
        )

        pred_face_coords = to_logits(decoded)

        return(
            recon_faces, # [B,F,3,3]
            loss
        )

class MeshTransformer:
    def generate(
        return_codes = False
    ):
        return codes # [B,F*3,2]
        face_coords = decode_from_codes_to_faces(codes) # [B,F,3,3]

        return files

    def forward(
        vertices, # [B,V,3]
        faces, # [B,F,3]
        codes = None
    ):
        return loss