#version 410
// One brush for the whole scroll.
//   kind 0: wash mesh, e=distance from wet edge. Pools at the edge, mottled, granulated by paper tooth.
//   kind 1: bristle ribbon. Each hair has its own ink; the brush runs dry along the stroke and
//           skips the paper's tooth (feibai, the same physics the camera's pan uses).
// out: r=ink density, g=reserve (how much of what is behind this mark is hidden), b=wetness.
in vec2 vP, vUV, vSD; in vec4 vA; in vec3 vB;
out vec4 o;
uniform sampler2D uNoise;   // r cloud, g mid, b tooth, a hair (uniform marginals, 1024 tile)
uniform float uTooth;

void main(){
  float kind=vB.x, occ=vB.y, wet=vB.z, seed=vA.w;
  float tooth=texture(uNoise,vP*uTooth).b;
  float D=0., al=0., W=0.;
  if(kind<.5){
    float e=vUV.x, pool=vUV.y, load=vA.x, mott=vA.y, soft=max(vA.z,.0015);
    float n0=texture(uNoise,vP*1.13+seed*vec2(.37,.71)).r;
    float n1=texture(uNoise,vP*3.1+seed*vec2(.13,.29)).g;
    float n2=texture(uNoise,vP*.37+seed*vec2(.53,.17)).r;
    float ee=e+(n0-.5)*soft*1.4+(n1-.5)*min(soft,.012);
    float m=smoothstep(0.,soft,ee);
    float rim=1.-smoothstep(soft*.15,soft+.012,ee);
    float mo=1.+mott*((n0-.5)*1.1+(n2-.5)*1.1+(n1-.5)*.35);
    D=load*m*max(mo,.05)*(1.+pool*rim)*(.90+.20*tooth);
    al=occ*m;
    W=wet*m;
  }else{
    float u=vUV.x, v=vUV.y, s=vSD.x, d=vSD.y;
    float load=vA.x, dry=vA.y, rough=vA.z;
    float wander=texture(uNoise,vec2(s*5.1+seed*91.,seed*17.+.3)).g-.5;
    float ev=abs(v)+wander*.55*rough;
    float afw=fwidth(v)*.9+.02;
    float m=1.-smoothstep(1.-afw,1.+afw*.3,ev);
    // hairs: tracks along the stroke, about .004 view units apart
    float hair=texture(uNoise,vec2(d*1.25+seed*53.,s*.11+seed*7.)).a;
    float streak=texture(uNoise,vec2(d*.42+seed*29.,s*.05+seed*3.)).g;
    float th=dry*(.16+1.02*pow(u,1.6));
    float grip=mix(hair*.72+streak*.28,tooth,.30);
    float cov=smoothstep(th-.09,th+.09,grip+.02);
    if(dry<=.001) cov=1.;
    float belly=mix(1.,.86+.3*streak,.7);
    float rimk=1.+.4*smoothstep(.35,.95,abs(v))*(1.-dry);
    D=load*m*cov*belly*rimk*(.92+.16*tooth);
    al=occ*m*mix(cov,1.,.25);
    W=wet*m*cov;
  }
  o=vec4(D,al,W,al);
}
