#version 410
// tent upsample of the glare chain, blended with the same-size down level
in vec2 uv; out vec4 o;
uniform sampler2D uTex,uBase;
uniform vec2 uPx;
uniform float uMix;
void main(){
  vec2 d=uPx;
  vec3 r=texture(uTex,uv+vec2(-d.x,-d.y)).rgb+texture(uTex,uv+vec2(d.x,-d.y)).rgb
        +texture(uTex,uv+vec2(-d.x,d.y)).rgb+texture(uTex,uv+vec2(d.x,d.y)).rgb
        +2.*(texture(uTex,uv+vec2(-d.x,0.)).rgb+texture(uTex,uv+vec2(d.x,0.)).rgb
            +texture(uTex,uv+vec2(0.,-d.y)).rgb+texture(uTex,uv+vec2(0.,d.y)).rgb)
        +4.*texture(uTex,uv).rgb;
  r/=16.;
  o=vec4(mix(texture(uBase,uv).rgb,r,uMix),1.);
}
